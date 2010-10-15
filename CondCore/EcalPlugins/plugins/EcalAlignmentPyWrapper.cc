#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondTools/Ecal/interface/EcalAlignmentXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TPave.h"
#include "TPaveStats.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

using namespace std;
namespace cond {

  template<>
  class ValueExtractor<Alignments>: public  BaseValueExtractor<Alignments> {
  public:

    typedef Alignments Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
    }
  private:

  };

  template<>
  string PayLoadInspector<Alignments>::dump() const {
    
    stringstream ss;
    EcalCondHeader header;
    ss << EcalAlignmentXMLTranslator::dumpXML(header,object());
    return ss.str();
  }
  
  template<>
  string PayLoadInspector<Alignments>::summary() const {

    stringstream ss;   
    ss << "   Id    x   y   z   phi   theta   psi "  << endl;
    for ( vector<AlignTransform>::const_iterator it = object().m_align.begin();
	  it != object().m_align.end(); it++ ) {
      ss << hex << (*it).rawId()
	 << " " << (*it).translation().x()
	 << " " << (*it).translation().y()
	 << " " << (*it).translation().z()
	 << " " << (*it).rotation().getPhi()
	 << " " << (*it).rotation().getTheta() 
	 << " " << (*it).rotation().getPsi() 
	 << endl;
    }
    return ss.str();
  }
  

  template<>
  string PayLoadInspector<Alignments>::plot(string const & filename,
					    string const &, 
					    vector<int> const&, 
					    vector<float> const& ) const {
    return filename;
  }
}

PYTHON_WRAPPER(Alignments,Alignments);
