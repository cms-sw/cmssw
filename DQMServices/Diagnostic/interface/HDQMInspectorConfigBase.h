#ifndef GUARD_HDQMInspectorConfigBase_h
#define GUARD_HDQMInspectorConfigBase_h


#include <string>
#include <vector>
#include <map>

/**
 * @author: M. De Mattia
 * @date: 17/6/2009
 *
 * The HDQMinspectorConfigBase is the base class defining the interface
 * for the classes used to pass detector specific information to the
 * HDQMInspector. <br>
 * The methods defined are:
 * - translateDetId: a pure virtual method that receives the DetId and
 * returns a string. <br>
 * - valueErrorMap: a method filling a vector<pair<string, string> >
 * to associate user defined values with the corresponding errors. This
 * is optional and by default it will return false. <br>
 * - computeIntegralList: fills a vector<string> with the list of
 * quantities for which also a summation over the runs is required. <br>
 * An example of these are histograms containing entries, so that the
 * cumulative number of analyzed entries will be returned. <br>
 * It returns false by default. <br>
 *
 * Each subdetector must derive from this class and pass it by pointer
 * to the HDQMInspector.
 */


class HDQMInspectorConfigBase
{
 public:
   HDQMInspectorConfigBase () {};
   virtual ~HDQMInspectorConfigBase () {};
  /// pure virtual method that convert a DetId to a string
  virtual std::string translateDetId( const uint32_t ) const = 0;
  /// fills a vector<pair<string, string> > associating values with the corresponding errors
  virtual bool valueErrorMap(std::vector<std::pair<std::string, std::string> > & valueErrorVector) const {return false;}
  /// fills the list of names of quantities for which a summation over the runs is required
  virtual bool computeIntegralList(std::vector<std::string> & computeIntegralVector) const {return false;}

  std::string getErrorForQuantity(std::string const& QuantityName)
  {
    // Return the error name for a quantity name given.  This is designed to be used for the
    // "user" input quantities

    for (std::map<std::string, std::string>::iterator It = fErrorMap.begin(); It != fErrorMap.end(); ++It) {
      if (QuantityName.find( It->first ) != std::string::npos) {
        return It->second;
      }
    }

    return "";
  }

 private:
  std::map<std::string, std::string> fErrorMap;
};


/*
 * valueErrorMap: don't I need a way to access what is input here in the HDQMI code??
 *  map should be vlist=>error right?
 *
 * computeIntegralList: need a way to access that as well.
 *
 */



#endif
