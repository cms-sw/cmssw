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
 * $Id$
 *
 ************************************************************/
 
#include <vector>
#include <ostream>


class LumiDetails {

  public:

    /// default constructor
    LumiDetails() {}

    /// set default constructor
    LumiDetails(const std::vector<double>& lumietsum,const std::vector<double>& lumietsumerr, 
		const std::vector<double>& lumietsumqual, const std::vector<double>& lumiocc,
		const std::vector<double>& lumioccerr,const std::vector<double>& lumioccqual):
      lumietsum_(lumietsum), lumietsumerr_(lumietsumerr), lumietsumqual_(lumietsumqual),
      lumiocc_(lumiocc), lumioccerr_(lumioccerr), lumioccqual_(lumioccqual)
    { }

    /// destructor
    ~LumiDetails() {}

    // BX definition: There are 3564 bunch crossing (BX) in each LHC orbit 
    // each event will occur at one of these BX. BX is defined to be the number of the
    // bunch crossing where this event occurred.

    double lumiEtSum(int bx) const { return lumietsum_.at(bx); }
    double lumiEtSumErr(int bx) const { return lumietsumerr_.at(bx); }
    double lumiEtSumQual(int bx) const { return lumietsumqual_.at(bx); }

    double lumiOcc(int bx) const { return lumiocc_.at(bx); }
    double lumiOccErr(int bx) const { return lumioccerr_.at(bx); }
    double lumiOccQual(int bx) const { return lumioccqual_.at(bx); }

    //get array of lumi values per BX based on EtSum method
    const std::vector<double>&  lumiEtSum() const {return lumietsum_;}
       
    //get array of lumi values errors per BX based on EtSum method
    const std::vector<double>&  lumiEtSumErr() const {return lumietsumerr_;}
       
    //get array of lumi values quality per BX based on EtSum method
    const std::vector<double>&  lumiEtSumQual() const {return lumietsumqual_;}
       
    //get array of lumi values per BX based on Occupancy method
    const std::vector<double>&  lumiOcc() const {return lumiocc_;}
       
    //get array of lumi values errors per BX based on Occupancy method
    const std::vector<double>&  lumiOccErr() const {return lumioccerr_;}
       
    //get array of lumi values errors per BX based on Occupancy method
    const std::vector<double>&  lumiOccQual() const {return lumioccqual_;}

  private :

    std::vector<double> lumietsum_;
    std::vector<double> lumietsumerr_;
    std::vector<double> lumietsumqual_;

    std::vector<double> lumiocc_;
    std::vector<double> lumioccerr_;
    std::vector<double> lumioccqual_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails);

#endif // DataFormats_Luminosity_LumiDetails_h
