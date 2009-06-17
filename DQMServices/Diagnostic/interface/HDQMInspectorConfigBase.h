#include <string>
#include <vector>

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

using namespace std;

class HDQMInspectorConfigBase
{
 public:
  /// pure virtual method that convert a DetId to a string
  virtual string translateDetId( const uint32_t detId ) const = 0;
  /// fills a vector<pair<string, string> > associating values with the corresponding errors
  virtual bool valueErrorMap(vector<pair<string, string> > & valueErrorVector) const {return false;}
  /// fills the list of names of quantities for which a summation over the runs is required
  virtual bool computeIntegralList(vector<string> & computeIntegralVector) const {return false;}
};
