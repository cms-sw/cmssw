
#ifndef TKStatus_H
#define TKStatus_H

/** \class TKStatus
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *
 */
// C++
#include <string>
#include <array>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include <fstream>


//
// class declaration
//

class TKStatus : public edm::one::EDAnalyzer<> {
 public:
  TKStatus( const edm::ParameterSet& );

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c) override ;

 private:

  void dumpTkDcsStatus(std::string const &, edm::RunNumber_t, std::array<bool,6> const&);

  std::string dcsTkFileName_;
  edm::EDGetTokenT<DcsStatusCollection> dcsStatus_;

  int  lastlumi_ = -1;
  // ----------member data ---------------------------


};

#endif


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
