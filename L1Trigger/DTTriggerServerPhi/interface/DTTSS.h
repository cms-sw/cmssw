//-------------------------------------------------
//
/**  \class DTTSS
 *    Implementation of TSS trigger algorithm
 *
 *
 *   $Date: 2008/09/05 15:59:57 $
 *   $Revision: 1.3 $
 *
 *   \author C. Grandi, D. Bonacorsi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_TSS_H
#define DT_TSS_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTTracoTrigData;
class DTTSCand;
class DTConfigTSPhi;

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTSS {

  public:

    /// Constructor
    DTTSS(int);
  
    /// Destructor 
    ~DTTSS();

    /// Add a TS candidate to the TSS, ifs is first/second track flag
    void addDTTSCand(DTTSCand* cand);

    /// Set configuration
    void setConfig(DTConfigTSPhi *config) {  _config=config; }

    /// Set a flag to skip sort2
    void ignoreSecondTrack() { _ignoreSecondTrack=1; }

    /// Run the TSS algorithm
    void run();

    /// Sort 1
    DTTSCand* sortTSS1();

    /// Sort 2
    DTTSCand* sortTSS2();

    /// Clear
    void clear();

    /// Return identifier
    inline int number() const { return _n; }

    /// Configuration set
    inline DTConfigTSPhi* config() const { return _config; }

    /// Return the number of input tracks (first/second)
    unsigned nTracoT(int ifs) const;

    /// Return the number of input first tracks
    inline int nFirstT() const { return _tctrig[0].size(); }

    /// Return the number of input second tracks
    inline int nSecondT() const { return _tctrig[1].size(); }

    /// Return requested TS candidate
    DTTSCand* getDTTSCand(int ifs, unsigned n) const;

    /// Return requested TRACO trigger
    const DTTracoTrigData* getTracoT(int ifs, unsigned n) const;

    /// Return the carry (for debugging)
    DTTSCand* getCarry() const;

    /// Return the number of sorted tracks
    inline int nTracks() const { return _outcand.size(); }

    /// Return the requested track
    DTTSCand* getTrack(int n) const;

    /// Return the requested log word
      std::string logWord(int n) const;

  private:

    DTConfigTSPhi* _config;

    // identification
    int _n;

    // input data
    std::vector<DTTSCand*> _tctrig[2];

    // output data
    std::vector<DTTSCand*> _outcand;

    // internal use variables
    int _ignoreSecondTrack;

    // log words
    std::string _logWord1;
    std::string _logWord2;
  
};

#endif
