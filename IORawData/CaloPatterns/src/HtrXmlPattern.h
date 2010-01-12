#ifndef HtrXmlPattern_h_included
#define HtrXmlPattern_h_included 1

// system include files
#include <memory>

// default include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

//additional include files
#include "HtrXmlPatternTool.h"
#include "HtrXmlPatternToolParameters.h"

class HtrXmlPattern : public edm::EDAnalyzer {
public:
  explicit HtrXmlPattern(const edm::ParameterSet&);
  ~HtrXmlPattern();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void do_hand_fill(const HcalElectronicsMap*);
  HtrXmlPatternTool *m_tool;
  HtrXmlPatternToolParameters *m_toolparameters;
  int  m_sets_to_show;
  int  m_hand_pattern_number;
  bool m_fill_by_hand;
  bool m_filled;
  bool m_write_root_file;
};

#endif
