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
 * $Id: LumiDetails.h,v 1.3 2007/07/24 21:48:35 wdd Exp $
 *
 ************************************************************/
 
#include <vector>
#include <ostream>


class LumiDetails {

  public:

    /// default constructor
    LumiDetails() {}

    /// set default constructor
    LumiDetails(const std::vector<float>& lumietsum, const std::vector<float>& lumietsumerr, 
		const std::vector<int>& lumietsumqual, const std::vector<float>& lumiocc,
		const std::vector<float>& lumioccerr, const std::vector<int>& lumioccqual):
      lumietsum_(lumietsum), lumietsumerr_(lumietsumerr), lumietsumqual_(lumietsumqual),
      lumiocc_(lumiocc), lumioccerr_(lumioccerr), lumioccqual_(lumioccqual)
    { }

    /// destructor
    ~LumiDetails() {}

    // BX definition: There are 3564 bunch crossing (BX) in each LHC orbit 
    // each event will occur at one of these BX. BX is defined to be the number of the
    // bunch crossing where this event occurred.

    float lumiEtSum(int bx) const { return lumietsum_.at(bx); }
    float lumiEtSumErr(int bx) const { return lumietsumerr_.at(bx); }
    int lumiEtSumQual(int bx) const { return lumietsumqual_.at(bx); }

    float lumiOcc(int bx) const { return lumiocc_.at(bx); }
    float lumiOccErr(int bx) const { return lumioccerr_.at(bx); }
    int lumiOccQual(int bx) const { return lumioccqual_.at(bx); }

    //get array of lumi values per BX based on EtSum method
    const std::vector<float>&  lumiEtSum() const {return lumietsum_;}
       
    //get array of lumi values errors per BX based on EtSum method
    const std::vector<float>&  lumiEtSumErr() const {return lumietsumerr_;}
       
    //get array of lumi values quality per BX based on EtSum method
    const std::vector<int>&  lumiEtSumQual() const {return lumietsumqual_;}
       
    //get array of lumi values per BX based on Occupancy method
    const std::vector<float>&  lumiOcc() const {return lumiocc_;}
       
    //get array of lumi values errors per BX based on Occupancy method
    const std::vector<float>&  lumiOccErr() const {return lumioccerr_;}
       
    //get array of lumi values errors per BX based on Occupancy method
    const std::vector<int>&  lumiOccQual() const {return lumioccqual_;}

  private :

    std::vector<float> lumietsum_;
    std::vector<float> lumietsumerr_;
    std::vector<int> lumietsumqual_;

    std::vector<float> lumiocc_;
    std::vector<float> lumioccerr_;
    std::vector<int> lumioccqual_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails);

#endif // DataFormats_Luminosity_LumiDetails_h
