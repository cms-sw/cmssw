#include "CondTools/Ecal/interface/EcalADCToGeVTagWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

popcon::EcalADCToGeVTagWriter::EcalADCToGeVTagWriter(const edm::ParameterSet& ps)
    : name_(ps.getUntrackedParameter<std::string>("name", "EcalADCToGeVTagWriter")),
      since_(ps.getParameter<unsigned int>("since")),
      ebval_(ps.getParameter<double>("ebConst")),
      eeval_(ps.getParameter<double>("eeConst")) {}

void popcon::EcalADCToGeVTagWriter::getNewObjects() {
  edm::LogInfo("EcalADCToGeVTagWriter") << "Writing tag payload for IOV " << since_ << ": EB constant: " << ebval_
                                        << ", EE constant: " << eeval_;
  EcalADCToGeVConstant* payload = new EcalADCToGeVConstant(ebval_, eeval_);
  popcon::PopConSourceHandler<EcalADCToGeVConstant>::m_to_transfer.push_back(
      std::make_pair(payload, static_cast<Time_t>(since_)));
}
