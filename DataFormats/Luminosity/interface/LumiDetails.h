#ifndef DataFormats_Luminosity_LumiDetails_h
#define DataFormats_Luminosity_LumiDetails_h
 
/** \class LumiDetails
 *
 *
 * LumiDetails holds Details information the lumi value, the error on this value 
 * and its quality for each bunch crossing (BX) in a given luminosity section (LS)   
 *
 * \author Valerie Halyo
 *         David Dagenhart
 *
 * \version   1st Version June 7 2007>
 * $Id: LumiDetails.h,v 1.6 2008/11/22 05:39:42 wmtan Exp $
 *
 ************************************************************/
 
#include <vector>
#include <map>
#include <string>
#include <ostream>

class LumiDetails {
  // BX definition: There are 3564 bunch crossing (BX) in each LHC orbit 
  // each event will occur at one of these BX. BX is defined to be the number of the bunch crossing where this event occurred.
 public:

  /// default constructor
  LumiDetails();
  
  /// 
  void swapValueData(std::map<std::string,std::vector<float> >& data);
  ///
  void swapErrorData(std::map<std::string,std::vector<float> >& data);
  ///
  void swapQualData(std::map<std::string,std::vector<short> >& data);
  ///
  void copyValueData(const std::map<std::string,std::vector<float> >& data);
  ///
  void copyErrorData(const std::map<std::string,std::vector<float> >& data);
  ///
  void copyQualData(const std::map<std::string,std::vector<short> >& data);
  /// destructor
  ~LumiDetails();
  
  float lumiValue(const std::string& algoname,unsigned int bx) const;
  float lumiError(const std::string& algoname,unsigned int bx) const;
  short lumiQuality(const std::string& algoname,unsigned int bx) const; 
  
  const std::vector<float>& lumiValuesForAlgo(const std::string& algoname) const;
  const std::vector<float>& lumiErrorsForAlgo(const std::string& algoname) const;    
  const std::vector<short>& lumiQualsForAlgo(const std::string& algoname) const;    
  const std::map< std::string,std::vector<float> >& allLumiValues()const;
  
  const std::map< std::string,std::vector<float> >& allLumiErrors()const;
  
  const std::map< std::string,std::vector<short> >& allLumiQuals()const;
  
  std::vector<std::string> algoNames()const;
  
  unsigned int totalLumiAlgos()const;
  
  
  //bool isProductEqual(LumiDetails const& next) const;??
  
 private :
  std::map< std::string,std::vector<float> > m_lumivalueMap; //algoname,vector of values in bx
  std::map< std::string,std::vector<float> > m_lumierrorMap; //algoname,vector of errors in bx
  std::map< std::string,std::vector<short> > m_lumiqualityMap; //algoname,vector of quality in bx
}; 

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails);

#endif // DataFormats_Luminosity_LumiDetails_h
