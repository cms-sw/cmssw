#ifndef CondCore_Utilities_PayloadInspector_h
#define CondCore_Utilities_PayloadInspector_h

#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/Session.h"
#include <iostream>

#include <string>
#include <tuple>
#include <vector>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>

namespace cond {

  namespace payloadInspector {

    // Metadata dictionary
    struct PlotAnnotations {
      static constexpr const char* const PLOT_TYPE_K = "type";
      static constexpr const char* const TITLE_K = "title";
      static constexpr const char* const PAYLOAD_TYPE_K = "payload_type";
      static constexpr const char* const INFO_K = "info";
      static constexpr const char* const XAXIS_K = "x_label";
      static constexpr const char* const YAXIS_K = "y_label";
      static constexpr const char* const ZAXIS_K = "z_label";
      PlotAnnotations();
      std::string get( const std::string& key ) const;
      std::map<std::string,std::string> m;
      bool singleIov = false;
    };

    static const char* const JSON_FORMAT_VERSION = "1.0";

    // Serialize functions 
    template <typename V> std::string serializeValue( const std::string& entryLabel, const V& value ){
      std::stringstream ss;
      ss << "\""<<entryLabel<<"\":"<<value;
      return ss.str();
    }

    template <> std::string serializeValue( const std::string& entryLabel, const std::string& value ){
      std::stringstream ss;
      ss << "\""<<entryLabel<<"\":\""<<value<<"\"";
      return ss.str();
    }

   // Specialization for the multi-values coordinates ( to support the combined time+runlumi abscissa )
    template <typename V> std::string serializeValue( const std::string& entryLabel, const std::tuple<V,std::string>& value ){
      std::stringstream ss;
      ss << serializeValue( entryLabel, std::get<0>(value) );
      ss << ", ";
      ss << serializeValue( entryLabel+"_label", std::get<1>(value) );
      return ss.str();
    }

    // Specialization for the error bars
    template <typename V> std::string serializeValue( const std::string& entryLabel, const std::pair<V,V>& value ){
      std::stringstream ss;
      ss << serializeValue( entryLabel, value.first );
      ss << ", ";
      ss << serializeValue( entryLabel+"_err", value.second );
      return ss.str();
    }

    std::string serializeAnnotations( const PlotAnnotations& annotations ){
      std::stringstream ss;
      ss <<"\"version\": \""<<JSON_FORMAT_VERSION<<"\",";
      ss <<"\"annotations\": {";
      bool first = true;
      for( auto a: annotations.m ){
        if ( !first ) ss <<",";
	ss << "\""<<a.first<<"\":\"" << a.second <<"\"";
        first = false;
      } 
      ss <<"}";
      return ss.str();
    }

    template <typename X,typename Y> std::string serialize( const PlotAnnotations& annotations, const std::vector<std::tuple<X,Y> >& data ){
      // prototype implementation...
      std::stringstream ss;
      ss << "{";
      ss << serializeAnnotations( annotations );
      ss <<",";
      ss << "\"data\": [";
      bool first = true;
      for ( auto d : data ){
        if( !first ) ss <<",";
        ss <<" { "<<serializeValue("x",std::get<0>(d))<<", "<<serializeValue( "y",std::get<1>( d))<<" }";
        first = false;
      }
      ss << "]";
      ss << "}";
      return ss.str();
    }

    template <typename X,typename Y,typename Z> std::string serialize( const PlotAnnotations& annotations, const std::vector<std::tuple<X,Y,Z> >& data ){
      // prototype implementation...
      std::stringstream ss;
      ss << "{";
      ss << serializeAnnotations( annotations );
      ss <<",";
      ss << "\"data\": [";
      bool first = true;
      for ( auto d : data ){
        if( !first ) ss <<",";
        ss <<" { "<<serializeValue("x",std::get<0>(d))<<", "<<serializeValue( "y",std::get<1>( d))<<", "<<serializeValue( "z",std::get<2>( d))<<" }";
        first = false;
      }
      ss << "]";
      ss << "}";
      return ss.str();
    }

    struct ModuleVersion {
      static constexpr const char* const label = "1.0";
    };

    // Base class, factorizing the functions exposed in the python interface
    class PlotBase {
    public:
      PlotBase();

      // required in the browser to find corresponding tags
      std::string payloadType() const;

      // required in the browser
      std::string title() const; 

      // required in the browser
      std::string type() const;

      // required in the browser
      bool isSingleIov() const;

      // returns the json file with the plot data
      std::string data() const;

      // triggers the processing producing the plot
      bool process( const std::string& connectionString, const std::string& tag, const std::string& timeType, cond::Time_t begin, cond::Time_t end );

      // not exposed in python:
      // called internally in process()
      virtual void init();

      // not exposed in python:
      // called internally in process()
      virtual std::string processData( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs );

      // to be set in order to limit the iov selection ( and processing ) to 1 iov
      void setSingleIov( bool flag );

      // access to the fetch function of the configured reader, to be used in the processData implementations
      template <typename PayloadType> std::shared_ptr<PayloadType> fetchPayload( const cond::Hash& payloadHash ){
	return m_dbSession.fetchPayload<PayloadType>( payloadHash );
      }

      // access to the timeType of the tag in process
      cond::TimeType tagTimeType() const;

      // access to the underlying db session
      cond::persistency::Session dbSession();

    protected:

      PlotAnnotations m_plotAnnotations;

    private:

      cond::persistency::Session m_dbSession;
      cond::TimeType m_tagTimeType;

      std::string m_data = "";
    };

    // Concrete plot-types implementations
    template <typename PayloadType, typename X, typename Y> class Plot2D : public PlotBase {
    public:
      Plot2D( const std::string& type, const std::string& title, const std::string xLabel, const std::string& yLabel ) : 
	PlotBase(),m_plotData(){
	m_plotAnnotations.m[PlotAnnotations::PLOT_TYPE_K] = type;
        m_plotAnnotations.m[PlotAnnotations::TITLE_K] = title;
	m_plotAnnotations.m[PlotAnnotations::XAXIS_K] = xLabel;
	m_plotAnnotations.m[PlotAnnotations::YAXIS_K] = yLabel;
        m_plotAnnotations.m[PlotAnnotations::PAYLOAD_TYPE_K] = cond::demangledName( typeid(PayloadType) );
      }

      std::string serializeData(){
	return serialize( m_plotAnnotations, m_plotData);
      }

      std::string processData( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	fill( iovs );
	return serializeData();
      }

      std::shared_ptr<PayloadType> fetchPayload( const cond::Hash& payloadHash ){
      	return PlotBase::fetchPayload<PayloadType>( payloadHash );
      }

      virtual bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) = 0;
    protected:
      std::vector<std::tuple<X,Y> > m_plotData;
    };

    template <typename PayloadType, typename X, typename Y, typename Z> class Plot3D : public PlotBase {
    public:
      Plot3D( const std::string& type, const std::string& title, const std::string xLabel, const std::string& yLabel, const std::string& zLabel ) :
        PlotBase(),m_plotData(){
        m_plotAnnotations.m[PlotAnnotations::PLOT_TYPE_K] = type;
        m_plotAnnotations.m[PlotAnnotations::TITLE_K] = title;
        m_plotAnnotations.m[PlotAnnotations::XAXIS_K] = xLabel;
        m_plotAnnotations.m[PlotAnnotations::YAXIS_K] = yLabel;
        m_plotAnnotations.m[PlotAnnotations::ZAXIS_K] = zLabel;
        m_plotAnnotations.m[PlotAnnotations::PAYLOAD_TYPE_K] = cond::demangledName( typeid(PayloadType) );
      }

      std::string serializeData(){
	return serialize( m_plotAnnotations, m_plotData);
      }

      std::string processData( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	fill( iovs );
	return serializeData();
      }

      std::shared_ptr<PayloadType> fetchPayload( const cond::Hash& payloadHash ){
      	return PlotBase::fetchPayload<PayloadType>( payloadHash );
      }

      virtual bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) = 0;
    protected:
      std::vector<std::tuple<X,Y,Z> > m_plotData;
    };

    template<typename PayloadType, typename Y> class HistoryPlot : public Plot2D<PayloadType,unsigned long long,Y > {
    public:
      typedef Plot2D<PayloadType,unsigned long long,Y > Base;

      HistoryPlot( const std::string& title, const std::string& yLabel ) : 
	Base( "History", title, "iov_since" , yLabel ){
      }

      bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	for( auto iov : iovs ) {
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
	    Y value = getFromPayload( *payload );
	    Base::m_plotData.push_back( std::make_tuple(std::get<0>(iov),value));
	  }
	}
	return true;
      }

      virtual Y getFromPayload( PayloadType& payload ) = 0; 

    };

    template<typename PayloadType, typename Y> class RunHistoryPlot : public Plot2D<PayloadType,std::tuple<float,std::string>,Y > {
    public:
      typedef Plot2D<PayloadType,std::tuple<float,std::string>,Y > Base;

      RunHistoryPlot( const std::string& title, const std::string& yLabel ) :
        Base( "RunHistory", title, "iov_since" , yLabel ){
      }

      bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
        // for the lumi iovs we need to count the number of lumisections in every runs
	std::map<cond::Time_t,unsigned int> runs;
	if( Base::tagTimeType()==cond::lumiid ){
	  for( auto iov : iovs ) {
            unsigned int run = std::get<0>(iov) >> 32;
	    auto it = runs.find( run );
            if( it == runs.end() ) it = runs.insert( std::make_pair( run, 0 ) ).first;
            it->second++;
	  }
	}		
        unsigned int currentRun = 0;
	float lumiIndex = 0;
	unsigned int lumiSize = 0;
        unsigned int rind = 0;
        float ind = 0;
	std::string label("");
        for( auto iov : iovs )  {
	  unsigned long long since = std::get<0>(iov);
          // for the lumi iovs we squeeze the lumi section available in the constant run 'slot' of witdth=1
          if( Base::tagTimeType()==cond::lumiid ){
	    unsigned int run = since >> 32;
            unsigned int lumi = since & 0xFFFFFFFF;
            if( run != currentRun ) {
              rind++;
	      lumiIndex = 0;
              auto it = runs.find( run );
	      if( it == runs.end() ) {
		// it should never happen
		return false;
	      }
              lumiSize = it->second;
	    } else {
	      lumiIndex++;
	    }
	    ind =  rind + (lumiIndex/lumiSize); 
            label = boost::lexical_cast<std::string>( run )+" : "+boost::lexical_cast<std::string>( lumi );
	    currentRun = run;
	  } else {
	    ind++;
	    // for the timestamp based iovs, it does not really make much sense to use this plot... 
            if(  Base::tagTimeType()==cond::timestamp ){
	      boost::posix_time::ptime t = cond::time::to_boost( since );
	      label = boost::posix_time::to_simple_string( t );
	    } else {
	      label = boost::lexical_cast<std::string>( since );
	    } 
	  }
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
            Y value = getFromPayload( *payload );
	    Base::m_plotData.push_back( std::make_tuple(std::make_tuple(ind,label),value));
          }
	}
        return true;
      }

      virtual Y getFromPayload( PayloadType& payload ) = 0;

    };

    template<typename PayloadType, typename Y> class TimeHistoryPlot : public Plot2D<PayloadType,std::tuple<unsigned long long,std::string>,Y > {
    public:
      typedef Plot2D<PayloadType,std::tuple<unsigned long long,std::string>,Y > Base;

      TimeHistoryPlot( const std::string& title, const std::string& yLabel ) : 
	Base( "TimeHistory", title, "iov_since" , yLabel ){
      }

      bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	cond::persistency::RunInfoProxy runInfo;
	if(  Base::tagTimeType()==cond::lumiid ||  Base::tagTimeType()==cond::runnumber){
	  cond::Time_t min = std::get<0>(iovs.front());
	  cond::Time_t max = std::get<0>( iovs.back() );
          if(  Base::tagTimeType()==cond::lumiid ){
	    min = min >> 32;
            max = max >> 32;
	  }
	  runInfo = Base::dbSession().getRunInfo( min, max );
	}
	for( auto iov : iovs ) {
	  cond::Time_t since = std::get<0>(iov);
	  boost::posix_time::ptime time;
	  std::string label("");
          if(  Base::tagTimeType()==cond::lumiid ||  Base::tagTimeType()==cond::runnumber){
            unsigned int nlumi = since & 0xFFFFFFFF;
	    if( Base::tagTimeType()==cond::lumiid ) since = since >> 32;
            label = boost::lexical_cast<std::string>( since );
	    auto it = runInfo.find( since );
	    if ( it == runInfo.end() ){
	      // this should never happen...
	      return false;
	    }
	    time = (*it).start;
	    // add the lumi sections...
	    if(  Base::tagTimeType()==cond::lumiid ){
	      time += boost::posix_time::seconds( cond::time::SECONDS_PER_LUMI*nlumi );
              label += (" : "+boost::lexical_cast<std::string>( nlumi ) );
	    }
	  } else if (  Base::tagTimeType()==cond::timestamp ){
	    time = cond::time::to_boost( since );
            label = boost::posix_time::to_simple_string( time );
	  }
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
	    Y value = getFromPayload( *payload );
	    Base::m_plotData.push_back( std::make_tuple(std::make_tuple( cond::time::from_boost(time),label),value));
	  }
	}
	return true;
      }

      virtual Y getFromPayload( PayloadType& payload ) = 0; 

    };

    template<typename PayloadType, typename X, typename Y> class ScatterPlot : public Plot2D<PayloadType,X,Y > {
    public:
      typedef Plot2D<PayloadType,X,Y > Base;
      // the x axis label will be overwritten by the plot rendering application
      ScatterPlot( const std::string& title, const std::string& xLabel, const std::string& yLabel ) :
	Base( "Scatter", title, xLabel , yLabel ){
      }

      bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	for( auto iov : iovs ) {
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
	    std::tuple<X,Y> value = getFromPayload( *payload );
	    Base::m_plotData.push_back( value );
	  }
	}
	return true;
      }

      virtual std::tuple<X,Y> getFromPayload( PayloadType& payload ) = 0; 

    };

    //
    template<typename PayloadType> class Histogram1D: public Plot2D<PayloadType,float,float > {
    public:
      typedef Plot2D<PayloadType,float,float > Base;
      // naive implementation, essentially provided as an example...
      Histogram1D( const std::string& title, const std::string& xLabel, size_t nbins, float min, float max ):
	Base( "Histo1D", title, xLabel , "entries" ),m_nbins(nbins),m_min(min),m_max(max){
      }

      // 
      void init(){
	Base::m_plotData.clear();
	float binSize = (m_max-m_min)/m_nbins;
	if( binSize>0 ){
          m_binSize = binSize;
	  Base::m_plotData.resize( m_nbins );
	  for( size_t i=0;i<m_nbins;i++ ){
	    Base::m_plotData[i] = std::make_tuple( m_min+i*m_binSize, 0 );
	  }
	}
      }

      // to be used to fill the histogram!
      void fillWithValue( float value, float weight=1 ){
	// ignoring underflow/overflows ( they can be easily added - the total entries as well  )
        if( Base::m_plotData.size() && (value < m_max) && (value >= m_min) ){
	  size_t ibin = (value-m_min)/m_binSize;
	  std::get<1>(Base::m_plotData[ibin])+=weight;
	}
      }

      // this one can ( and in general should ) be overridden - the implementation should use fillWithValue
      virtual bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	for( auto iov : iovs ) {
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
	    float value = getFromPayload( *payload );
	    fillWithValue( value );
	  }
        }
	return true;
      }

      // implement this one if you use the default fill implementation, otherwise ignore it...
      virtual float getFromPayload( PayloadType& payload ) {
	return 0;
      }

    private:
      float m_binSize = 0;
      size_t m_nbins; 
      float m_min;
      float m_max;
    };

    // 
    template<typename PayloadType> class Histogram2D: public Plot3D<PayloadType,float,float,float > {
    public:
      typedef Plot3D<PayloadType,float,float,float > Base;
      // naive implementation, essentially provided as an example...
      Histogram2D( const std::string& title, const std::string& xLabel, size_t nxbins, float xmin, float xmax, 
		                             const std::string& yLabel, size_t nybins, float ymin, float ymax  ):
	Base( "Histo2D", title, xLabel , yLabel, "entries" ),m_nxbins( nxbins), m_xmin(xmin),m_xmax(xmax),m_nybins(nybins),m_ymin(ymin),m_ymax(ymax){
      }

      //
      void init(){
	Base::m_plotData.clear();
	float xbinSize = (m_xmax-m_xmin)/m_nxbins;
        float ybinSize = (m_ymax-m_ymin)/m_nybins;
	if( xbinSize>0 && ybinSize>0){
          m_xbinSize = xbinSize;
          m_ybinSize = ybinSize;
	  Base::m_plotData.resize( m_nxbins*m_nybins );
	  for( size_t i=0;i<m_nybins;i++ ){
	    for( size_t j=0;j<m_nxbins;j++ ){
	      Base::m_plotData[i*m_nxbins+j] = std::make_tuple( m_xmin+j*m_xbinSize, m_ymin+i*m_ybinSize, 0 );
	    }
	  }
	}
      }

      // to be used to fill the histogram!
      void fillWithValue( float xvalue, float yvalue, float weight=1 ){
	// ignoring underflow/overflows ( they can be easily added - the total entries as well )
	if( Base::m_plotData.size() && xvalue < m_xmax && xvalue >= m_xmin &&  yvalue < m_ymax && yvalue >= m_ymin ){
	  size_t ixbin = (xvalue-m_xmin)/m_xbinSize;
	  size_t iybin = (yvalue-m_ymin)/m_ybinSize;
	  std::get<2>(Base::m_plotData[iybin*m_nxbins+ixbin])+=weight;
	}
      }

      // this one can ( and in general should ) be overridden - the implementation should use fillWithValue
      virtual bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override {
	for( auto iov : iovs ) {
	  std::shared_ptr<PayloadType> payload = Base::fetchPayload( std::get<1>(iov) );
          if( payload.get() ){
	    std::tuple<float,float> value = getFromPayload( *payload );
	    fillWithValue( std::get<0>(value),std::get<1>(value) );
	  }
        }
	return true;
      }

      // implement this one if you use the default fill implementation, otherwise ignore it...
      virtual std::tuple<float,float> getFromPayload( PayloadType& payload ) {
        float x = 0;
	float y = 0;
	return std::make_tuple(x,y);
      }

    private:
      size_t m_nxbins;
      float m_xbinSize = 0;   
      float m_xmin;
      float m_xmax;
      float m_ybinSize = 0;
      size_t m_nybins; 
      float m_ymin;
      float m_ymax;
    };

  }

}

#endif

