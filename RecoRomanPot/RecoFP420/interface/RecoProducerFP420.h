#ifndef RecoProducerFP420_h
#define RecoProducerFP420_h

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FP420Cluster/interface/RecoFP420.h"

//Hector headers
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"
#include <string>


class RecoProducerFP420 {

public:
  RecoProducerFP420(const edm::ParameterSet& conf);
  //  Hector();
  virtual ~RecoProducerFP420();

  std::vector<RecoFP420> reconstruct(int,double,double,double,double,double,double);

private:
  edm::ParameterSet conf_;
  std::vector<RecoFP420> rhits; 

  int verbosity;

  double m_x0, m_y0, m_q2, m_e;
  double m_tx0, m_ty0;

  // Hector
  H_BeamLine * m_beamlineFP4201;
  H_BeamLine * m_beamlineFP4202;
  double lengthfp420;
  H_BeamLine * m_beamlineHPS2401;
  H_BeamLine * m_beamlineHPS2402;
  double lengthhps240;
  std::string beam1filename;
  std::string beam2filename;

  //
  H_RecRPObject * m_rp420_f;
  H_RecRPObject * m_rp420_b;
  H_RecRPObject * m_rp240_f;
  H_RecRPObject * m_rp240_b;

};
#endif
