#ifndef DQMOffline_RecoB_TrackIPHistograms_h
#define DQMOffline_RecoB_TrackIPHistograms_h

#include <string>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"

template<class T>
class TrackIPHistograms : public FlavourHistograms<T>
{
  public:

  TrackIPHistograms(const std::string& baseNameTitle_ , const std::string& baseNameDescription_,
                    const int& nBins_, const double& lowerBound_, const double& upperBound_,
                    const bool& statistics, const bool& plotLog_, const bool& plotNormalized_,
                    const std::string& plotFirst_, const bool& update, const std::string& folder, 
		    const unsigned int& mc, const bool& quality, DQMStore::IBooker & ibook);

  virtual ~TrackIPHistograms(){};

  void fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T& variable, const bool& hasTrack) const;
  void fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T& variable, const bool& hasTrack, const T & w) const;
 
  void fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T* variable, const bool& hasTrack) const;
  void fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T* variable, const bool& hasTrack, const T & w) const;

  void settitle(const char* title);

  protected:

  void fillVariable ( const reco::TrackBase::TrackQuality& qual, const T & var, const bool& hasTrack) const;
  void fillVariable ( const reco::TrackBase::TrackQuality& qual, const T & var, const bool& hasTrack, const T & w) const;

  bool quality_;

  MonitorElement *theQual_undefined;
  MonitorElement *theQual_loose;
  MonitorElement *theQual_tight;
  MonitorElement *theQual_highpur;

  private:
  TrackIPHistograms(){}
};

template <class T>
TrackIPHistograms<T>::TrackIPHistograms (const std::string& baseNameTitle_, const std::string& baseNameDescription_,
                                         const int& nBins_, const double& lowerBound_, const double& upperBound_,
                                         const bool& statistics_, const bool& plotLog_, const bool& plotNormalized_,
                                         const std::string& plotFirst_, const bool& update, const std::string& folder, 
					 const unsigned int& mc, const bool& quality, DQMStore::IBooker & ibook) :
  FlavourHistograms<T>(baseNameTitle_, baseNameDescription_, nBins_, lowerBound_, upperBound_, statistics_, plotLog_, plotNormalized_,
                       plotFirst_, update, folder, mc, ibook), quality_(quality)
{
  if(quality_) {
    if(!update) {
      HistoProviderDQM prov("Btag",folder,ibook);
      theQual_undefined = prov.book1D( baseNameTitle_ + "QualUnDef" , baseNameDescription_ + " Undefined Quality", nBins_, lowerBound_, upperBound_);
      theQual_loose = prov.book1D( baseNameTitle_ + "QualLoose" , baseNameDescription_ + " Loose Quality", nBins_, lowerBound_, upperBound_);
      theQual_tight = prov.book1D( baseNameTitle_ + "QualTight" , baseNameDescription_ + " Tight Quality", nBins_, lowerBound_, upperBound_);
      theQual_highpur = prov.book1D( baseNameTitle_ + "QualHighPur" , baseNameDescription_ + " High Purity Quality", nBins_, lowerBound_, upperBound_);

      if( statistics_ ) {
        theQual_undefined->getTH1F()->Sumw2();
        theQual_loose->getTH1F()->Sumw2();
        theQual_tight->getTH1F()->Sumw2();
        theQual_highpur->getTH1F()->Sumw2();
      }
    } else {
      //is it useful? anyway access function is deprecated...
      HistoProviderDQM prov("Btag",folder,ibook);
      theQual_undefined = prov.access(baseNameTitle_ + "QualUnDef");
      theQual_loose = prov.access(baseNameTitle_ + "QualLoose");
      theQual_tight = prov.access(baseNameTitle_ + "QualTight");
      theQual_highpur = prov.access(baseNameTitle_ + "QualHighPur");
    }
  }
}

template <class T>
void TrackIPHistograms<T>::fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T& variable, const bool& hasTrack) const
{
  FlavourHistograms<T>::fill(flavour, variable);
  if(quality_)
    fillVariable(quality, variable, hasTrack);
}

template <class T>
void TrackIPHistograms<T>::fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T& variable, const bool& hasTrack, const T & w) const
{
  FlavourHistograms<T>::fill(flavour, variable , w);
  if(quality_)
    fillVariable(quality, variable, hasTrack, w);
}

template <class T>
void TrackIPHistograms<T>::fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T* variable, const bool& hasTrack) const
{
  const int* theArrayDimension = FlavourHistograms<T>::arrayDimension();
  const int& theMaxDimension = FlavourHistograms<T>::maxDimension();
  const int& theIndexToPlot = FlavourHistograms<T>::indexToPlot();

  FlavourHistograms<T>::fill(flavour, variable);
  if( theArrayDimension == 0 && quality_) {
    fillVariable( quality, *variable);
  } else {
      int iMax = (*theArrayDimension > theMaxDimension) ? theMaxDimension : *theArrayDimension ;
      for(int i = 0; i != iMax; ++i) {
        if( quality_ && (( theIndexToPlot < 0) || ( i == theIndexToPlot)) ) {
          fillVariable ( flavour , *(variable + i), hasTrack);
        }
      }

      if(theIndexToPlot >= iMax && quality_) {
        const T& theZero = static_cast<T> (0.0);
        fillVariable ( quality, theZero, hasTrack);
      }
  }
}

template <class T>
void TrackIPHistograms<T>::fill(const int& flavour, const reco::TrackBase::TrackQuality& quality, const T* variable, const bool& hasTrack, const T & w) const
{
  const int* theArrayDimension = FlavourHistograms<T>::arrayDimension();
  const int& theMaxDimension = FlavourHistograms<T>::maxDimension();
  const int& theIndexToPlot = FlavourHistograms<T>::indexToPlot();

  FlavourHistograms<T>::fill(flavour, variable ,w);
  if( theArrayDimension == 0 && quality_) {
    fillVariable( quality, *variable,w);
  } else {
      int iMax = (*theArrayDimension > theMaxDimension) ? theMaxDimension : *theArrayDimension ;
      for(int i = 0; i != iMax; ++i) {
        if( quality_ && (( theIndexToPlot < 0) || ( i == theIndexToPlot)) ) {
          fillVariable ( flavour , *(variable + i), hasTrack,w);
        }
      }

      if(theIndexToPlot >= iMax && quality_) {
        const T& theZero = static_cast<T> (0.0);
        fillVariable ( quality, theZero, hasTrack,w);
      }
  }
}

template <class T>
void TrackIPHistograms<T>::settitle(const char* title)
{
  FlavourHistograms<T>::settitle(title);
  theQual_undefined->setAxisTitle(title);
  theQual_loose->setAxisTitle(title);
  theQual_tight->setAxisTitle(title);
  theQual_highpur->setAxisTitle(title);
}

template<class T>
void TrackIPHistograms<T>::fillVariable( const reco::TrackBase::TrackQuality& qual, const T& var, const bool& hasTrack) const
{
  if(!hasTrack || !quality_) return;

  switch(qual) {
    case reco::TrackBase::loose:
      theQual_loose->Fill(var);
      break;
    case reco::TrackBase::tight:
      theQual_tight->Fill(var);
      break;
    case reco::TrackBase::highPurity:
      theQual_highpur->Fill(var);
      break;
    default:
      theQual_undefined->Fill(var);
      break;
  }
}

template<class T>
void TrackIPHistograms<T>::fillVariable( const reco::TrackBase::TrackQuality& qual, const T& var, const bool& hasTrack, const T & w) const
{
  if(!hasTrack || !quality_) return;

  switch(qual) {
    case reco::TrackBase::loose:
      theQual_loose->Fill(var,w);
      break;
    case reco::TrackBase::tight:
      theQual_tight->Fill(var,w);
      break;
    case reco::TrackBase::highPurity:
      theQual_highpur->Fill(var,w);
      break;
    default:
      theQual_undefined->Fill(var,w);
      break;
  }
}

#endif
