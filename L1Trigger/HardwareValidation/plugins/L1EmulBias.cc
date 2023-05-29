#include "L1Trigger/HardwareValidation/plugins/L1EmulBias.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

using namespace dedefs;

L1EmulBias::L1EmulBias(const edm::ParameterSet& iConfig) {
  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag", 0);

  for (int sys = 0; sys < DEnsys; sys++) {
    std::string label = SystLabel[sys] + "source";
    m_DEsource[sys][0] = iConfig.getParameter<edm::InputTag>(label);
    if (sys == CTF) {
      std::string label = "CTTsource";
      m_DEsource[sys][1] = iConfig.getParameter<edm::InputTag>(label);
    }
  }

  std::vector<unsigned int> compColls = iConfig.getUntrackedParameter<std::vector<unsigned int> >("DO_SYSTEM");
  for (int i = 0; i < DEnsys; i++)
    m_doSys[i] = compColls[i];

  if (verbose()) {
    std::cout << "[L1EmulBias] do sys? ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << m_doSys[i];
    std::cout << "\n\t";
    for (int i = 0; i < DEnsys; i++)
      if (m_doSys[i])
        std::cout << SystLabel[i] << " ";
    std::cout << std::endl;
  }

  std::string CollInstName[DEnsys][5];
  for (int i = 0; i < DEnsys; i++)
    for (int j = 0; j < 5; j++)
      CollInstName[i][j] = std::string("");

  CollInstName[GCT][0] += "isoEm";
  CollInstName[GCT][1] += "nonIsoEm";
  CollInstName[GCT][2] += "cenJets";
  CollInstName[GCT][3] += "forJets";
  CollInstName[GCT][4] += "tauJets";
  CollInstName[DTF][0] += "DT";
  CollInstName[DTF][1] += "DTTF";
  CollInstName[CTF][0] += "CSC";
  CollInstName[CTF][1] += "";
  CollInstName[RPC][0] += "RPCb";
  CollInstName[RPC][1] += "RPCf";

  for (int i = 0; i < DEnsys; i++)
    for (int j = 0; j < 5; j++)
      instName[i][j] = CollInstName[i][j];

  if (verbose())
    for (int i = 0; i < DEnsys; i++)
      for (int j = 0; j < 5; j++)
        if (!instName[i][j].empty())
          std::cout << "[emulbias] " << i << " " << SystLabel[i] << " " << j << " " << instName[i][j] << std::endl;

  ///assertion/temporary
  assert(ETP == 0);
  assert(HTP == 1);
  assert(RCT == 2);
  assert(GCT == 3);
  assert(DTP == 4);
  assert(DTF == 5);
  assert(CTP == 6);
  assert(CTF == 7);
  assert(RPC == 8);
  assert(LTC == 9);
  assert(GMT == 10);
  assert(GLT == 11);

  ///List of collections to be produced
  if (m_doSys[ETP])
    produces<EcalTrigPrimDigiCollection>(instName[ETP][0]);
  if (m_doSys[HTP])
    produces<HcalTrigPrimDigiCollection>(instName[HTP][0]);
  if (m_doSys[RCT])
    produces<L1CaloEmCollection>(instName[RCT][0]);
  if (m_doSys[RCT])
    produces<L1CaloRegionCollection>(instName[RCT][0]);
  if (m_doSys[GCT])
    produces<L1GctEmCandCollection>(instName[GCT][0]);
  if (m_doSys[GCT])
    produces<L1GctEmCandCollection>(instName[GCT][1]);
  if (m_doSys[GCT])
    produces<L1GctJetCandCollection>(instName[GCT][2]);
  if (m_doSys[GCT])
    produces<L1GctJetCandCollection>(instName[GCT][3]);
  if (m_doSys[GCT])
    produces<L1GctJetCandCollection>(instName[GCT][4]);
  if (m_doSys[DTP])
    produces<L1MuDTChambPhContainer>(instName[DTP][0]);
  if (m_doSys[DTP])
    produces<L1MuDTChambThContainer>(instName[DTP][0]);
  if (m_doSys[DTF])
    produces<L1MuRegionalCandCollection>(instName[DTF][0]);
  if (m_doSys[DTF])
    produces<L1MuDTTrackContainer>(instName[DTF][1]);
  if (m_doSys[CTP])
    produces<CSCCorrelatedLCTDigiCollection>(instName[CTP][0]);
  if (m_doSys[CTF])
    produces<L1MuRegionalCandCollection>(instName[CTF][0]);
  if (m_doSys[CTF])
    produces<L1CSCTrackCollection>(instName[CTF][1]);
  if (m_doSys[RPC])
    produces<L1MuRegionalCandCollection>(instName[RPC][0]);
  if (m_doSys[RPC])
    produces<L1MuRegionalCandCollection>(instName[RPC][1]);
  if (m_doSys[LTC])
    produces<LTCDigiCollection>(instName[LTC][0]);
  if (m_doSys[GMT])
    produces<L1MuGMTCandCollection>(instName[GMT][0]);
  if (m_doSys[GMT])
    produces<L1MuGMTReadoutCollection>(instName[GMT][0]);
  if (m_doSys[GLT])
    produces<L1GlobalTriggerReadoutRecord>(instName[GLT][0]);
  if (m_doSys[GLT])
    produces<L1GlobalTriggerEvmReadoutRecord>(instName[GLT][0]);
  if (m_doSys[GLT])
    produces<L1GlobalTriggerObjectMapRecord>(instName[GLT][0]);

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "L1EmulBias requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }

  if (verbose())
    std::cout << "L1EmulBias::L1EmulBias()... done." << std::endl;
}

L1EmulBias::~L1EmulBias() {}

// ------------ method called to produce the data  ------------
void L1EmulBias::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());

  if (verbose())
    std::cout << "L1EmulBias::produce...\n" << std::flush;

  /// list the emulated collections
  edm::Handle<EcalTrigPrimDigiCollection> ecal_tp_emul;
  edm::Handle<HcalTrigPrimDigiCollection> hcal_tp_emul;
  edm::Handle<L1CaloEmCollection> rct_em_emul;
  edm::Handle<L1CaloRegionCollection> rct_rgn_emul;
  edm::Handle<L1GctEmCandCollection> gct_isolaem_emul;
  edm::Handle<L1GctEmCandCollection> gct_noisoem_emul;
  edm::Handle<L1GctJetCandCollection> gct_cenjets_emul;
  edm::Handle<L1GctJetCandCollection> gct_forjets_emul;
  edm::Handle<L1GctJetCandCollection> gct_taujets_emul;
  edm::Handle<L1MuDTChambPhContainer> dtp_ph_emul;
  edm::Handle<L1MuDTChambThContainer> dtp_th_emul;
  edm::Handle<L1MuRegionalCandCollection> dtf_emul;
  edm::Handle<L1MuDTTrackContainer> dtf_trk_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> ctp_emul;
  edm::Handle<L1MuRegionalCandCollection> ctf_emul;
  edm::Handle<L1CSCTrackCollection> ctf_trk_emul;
  edm::Handle<L1MuRegionalCandCollection> rpc_cen_emul;
  edm::Handle<L1MuRegionalCandCollection> rpc_for_emul;
  edm::Handle<LTCDigiCollection> ltc_emul;
  edm::Handle<L1MuGMTCandCollection> gmt_emul;
  edm::Handle<L1MuGMTReadoutCollection> gmt_rdt_emul;
  edm::Handle<L1GlobalTriggerReadoutRecord> gt_em_emul;
  edm::Handle<L1GlobalTriggerReadoutRecord> glt_rdt_emul;
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> glt_evm_emul;
  edm::Handle<L1GlobalTriggerObjectMapRecord> glt_obj_emul;

  /// get the emulated collections
  if (m_doSys[ETP])
    iEvent.getByLabel(m_DEsource[ETP][0].label(), instName[ETP][0], ecal_tp_emul);
  if (m_doSys[HTP])
    iEvent.getByLabel(m_DEsource[HTP][0].label(), instName[HTP][0], hcal_tp_emul);
  if (m_doSys[RCT])
    iEvent.getByLabel(m_DEsource[RCT][0].label(), instName[RCT][0], rct_em_emul);
  if (m_doSys[RCT])
    iEvent.getByLabel(m_DEsource[RCT][0].label(), instName[RCT][0], rct_rgn_emul);
  if (m_doSys[GCT])
    iEvent.getByLabel(m_DEsource[GCT][0].label(), instName[GCT][0], gct_isolaem_emul);
  if (m_doSys[GCT])
    iEvent.getByLabel(m_DEsource[GCT][0].label(), instName[GCT][1], gct_noisoem_emul);
  if (m_doSys[GCT])
    iEvent.getByLabel(m_DEsource[GCT][0].label(), instName[GCT][2], gct_cenjets_emul);
  if (m_doSys[GCT])
    iEvent.getByLabel(m_DEsource[GCT][0].label(), instName[GCT][3], gct_forjets_emul);
  if (m_doSys[GCT])
    iEvent.getByLabel(m_DEsource[GCT][0].label(), instName[GCT][4], gct_taujets_emul);
  if (m_doSys[DTP])
    iEvent.getByLabel(m_DEsource[DTP][0].label(), instName[DTP][0], dtp_ph_emul);
  if (m_doSys[DTP])
    iEvent.getByLabel(m_DEsource[DTP][0].label(), instName[DTP][0], dtp_th_emul);
  if (m_doSys[DTF])
    iEvent.getByLabel(m_DEsource[DTF][0].label(), instName[DTF][0], dtf_emul);
  if (m_doSys[DTF])
    iEvent.getByLabel(m_DEsource[DTF][0].label(), instName[DTF][1], dtf_trk_emul);
  if (m_doSys[CTP])
    iEvent.getByLabel(m_DEsource[CTP][0].label(), instName[CTP][0], ctp_emul);
  if (m_doSys[CTF])
    iEvent.getByLabel(m_DEsource[CTF][0].label(), instName[CTF][0], ctf_emul);
  if (m_doSys[CTF])
    iEvent.getByLabel(m_DEsource[CTF][1].label(), instName[CTF][1], ctf_trk_emul);
  if (m_doSys[RPC])
    iEvent.getByLabel(m_DEsource[RPC][0].label(), instName[RPC][0], rpc_cen_emul);
  if (m_doSys[RPC])
    iEvent.getByLabel(m_DEsource[RPC][0].label(), instName[RPC][1], rpc_for_emul);
  if (m_doSys[LTC])
    iEvent.getByLabel(m_DEsource[LTC][0].label(), instName[LTC][0], ltc_emul);
  if (m_doSys[GMT])
    iEvent.getByLabel(m_DEsource[GMT][0].label(), instName[GMT][0], gmt_emul);
  if (m_doSys[GMT])
    iEvent.getByLabel(m_DEsource[GMT][0].label(), instName[GMT][0], gmt_rdt_emul);
  if (m_doSys[GLT])
    iEvent.getByLabel(m_DEsource[GLT][0].label(), instName[GLT][0], glt_rdt_emul);
  if (m_doSys[GLT])
    iEvent.getByLabel(m_DEsource[GLT][0].label(), instName[GLT][0], glt_evm_emul);
  if (m_doSys[GLT])
    iEvent.getByLabel(m_DEsource[GLT][0].label(), instName[GLT][0], glt_obj_emul);

  /// assert collection validity
  if (m_doSys[ETP])
    assert(ecal_tp_emul.isValid());
  if (m_doSys[HTP])
    assert(hcal_tp_emul.isValid());
  if (m_doSys[RCT])
    assert(rct_em_emul.isValid());
  if (m_doSys[RCT])
    assert(rct_rgn_emul.isValid());
  if (m_doSys[GCT])
    assert(gct_isolaem_emul.isValid());
  if (m_doSys[GCT])
    assert(gct_noisoem_emul.isValid());
  if (m_doSys[GCT])
    assert(gct_cenjets_emul.isValid());
  if (m_doSys[GCT])
    assert(gct_forjets_emul.isValid());
  if (m_doSys[GCT])
    assert(gct_taujets_emul.isValid());
  if (m_doSys[DTP])
    assert(dtp_ph_emul.isValid());
  if (m_doSys[DTP])
    assert(dtp_th_emul.isValid());
  if (m_doSys[DTF])
    assert(dtf_emul.isValid());
  if (m_doSys[DTF])
    assert(dtf_trk_emul.isValid());
  if (m_doSys[CTP])
    assert(ctp_emul.isValid());
  if (m_doSys[CTF])
    assert(ctf_emul.isValid());
  if (m_doSys[CTF])
    assert(ctf_trk_emul.isValid());
  if (m_doSys[RPC])
    assert(rpc_cen_emul.isValid());
  if (m_doSys[RPC])
    assert(rpc_for_emul.isValid());
  if (m_doSys[LTC])
    assert(ltc_emul.isValid());
  if (m_doSys[GMT])
    assert(gmt_emul.isValid());
  if (m_doSys[GMT])
    assert(gmt_rdt_emul.isValid());
  if (m_doSys[GLT])
    assert(glt_rdt_emul.isValid());
  if (m_doSys[GLT])
    assert(glt_evm_emul.isValid());
  if (m_doSys[GLT])
    assert(glt_obj_emul.isValid());

  /// declare the data collections
  std::unique_ptr<EcalTrigPrimDigiCollection> ecal_tp_data(new EcalTrigPrimDigiCollection);
  std::unique_ptr<HcalTrigPrimDigiCollection> hcal_tp_data(new HcalTrigPrimDigiCollection);
  std::unique_ptr<L1CaloEmCollection> rct_em_data(new L1CaloEmCollection);
  std::unique_ptr<L1CaloRegionCollection> rct_rgn_data(new L1CaloRegionCollection);
  std::unique_ptr<L1GctEmCandCollection> gct_isolaem_data(new L1GctEmCandCollection);
  std::unique_ptr<L1GctEmCandCollection> gct_noisoem_data(new L1GctEmCandCollection);
  std::unique_ptr<L1GctJetCandCollection> gct_cenjets_data(new L1GctJetCandCollection);
  std::unique_ptr<L1GctJetCandCollection> gct_forjets_data(new L1GctJetCandCollection);
  std::unique_ptr<L1GctJetCandCollection> gct_taujets_data(new L1GctJetCandCollection);
  std::unique_ptr<L1MuDTChambPhContainer> dtp_ph_data(new L1MuDTChambPhContainer);
  std::unique_ptr<L1MuDTChambThContainer> dtp_th_data(new L1MuDTChambThContainer);
  std::unique_ptr<L1MuRegionalCandCollection> dtf_data(new L1MuRegionalCandCollection);
  std::unique_ptr<L1MuDTTrackContainer> dtf_trk_data(new L1MuDTTrackContainer);
  std::unique_ptr<CSCCorrelatedLCTDigiCollection> ctp_data(new CSCCorrelatedLCTDigiCollection);
  std::unique_ptr<L1MuRegionalCandCollection> ctf_data(new L1MuRegionalCandCollection);
  std::unique_ptr<L1CSCTrackCollection> ctf_trk_data(new L1CSCTrackCollection);
  std::unique_ptr<L1MuRegionalCandCollection> rpc_cen_data(new L1MuRegionalCandCollection);
  std::unique_ptr<L1MuRegionalCandCollection> rpc_for_data(new L1MuRegionalCandCollection);
  std::unique_ptr<LTCDigiCollection> ltc_data(new LTCDigiCollection);
  std::unique_ptr<L1MuGMTCandCollection> gmt_data(new L1MuGMTCandCollection);
  std::unique_ptr<L1MuGMTReadoutCollection> gmt_rdt_data(new L1MuGMTReadoutCollection);
  std::unique_ptr<L1GlobalTriggerReadoutRecord> glt_rdt_data(new L1GlobalTriggerReadoutRecord);
  std::unique_ptr<L1GlobalTriggerEvmReadoutRecord> glt_evm_data(new L1GlobalTriggerEvmReadoutRecord);
  std::unique_ptr<L1GlobalTriggerObjectMapRecord> glt_obj_data(new L1GlobalTriggerObjectMapRecord);

  if (verbose())
    std::cout << "L1EmulBias::produce - modify...\n" << std::flush;

  /// fill data as modified emul collections
  if (m_doSys[ETP])
    ModifyCollection(ecal_tp_data, ecal_tp_emul, engine);
  if (m_doSys[HTP])
    ModifyCollection(hcal_tp_data, hcal_tp_emul, engine);
  if (m_doSys[RCT])
    ModifyCollection(rct_em_data, rct_em_emul, engine);
  if (m_doSys[RCT])
    ModifyCollection(rct_rgn_data, rct_rgn_emul, engine);
  if (m_doSys[GCT])
    ModifyCollection(gct_isolaem_data, gct_isolaem_emul, engine);
  if (m_doSys[GCT])
    ModifyCollection(gct_noisoem_data, gct_noisoem_emul, engine);
  if (m_doSys[GCT])
    ModifyCollection(gct_cenjets_data, gct_cenjets_emul, engine);
  if (m_doSys[GCT])
    ModifyCollection(gct_forjets_data, gct_forjets_emul, engine);
  if (m_doSys[GCT])
    ModifyCollection(gct_taujets_data, gct_taujets_emul, engine);
  if (m_doSys[DTP])
    ModifyCollection(dtp_ph_data, dtp_ph_emul, engine);
  if (m_doSys[DTP])
    ModifyCollection(dtp_th_data, dtp_th_emul, engine);
  if (m_doSys[DTF])
    ModifyCollection(dtf_data, dtf_emul, engine);
  if (m_doSys[DTF])
    ModifyCollection(dtf_trk_data, dtf_trk_emul, engine);
  if (m_doSys[CTP])
    ModifyCollection(ctp_data, ctp_emul, engine);
  if (m_doSys[CTF])
    ModifyCollection(ctf_data, ctf_emul, engine);
  if (m_doSys[CTF])
    ModifyCollection(ctf_trk_data, ctf_trk_emul, engine);
  if (m_doSys[RPC])
    ModifyCollection(rpc_cen_data, rpc_cen_emul, engine);
  if (m_doSys[RPC])
    ModifyCollection(rpc_for_data, rpc_for_emul, engine);
  if (m_doSys[LTC])
    ModifyCollection(ltc_data, ltc_emul, engine);
  if (m_doSys[GMT])
    ModifyCollection(gmt_data, gmt_emul, engine);
  if (m_doSys[GMT])
    ModifyCollection(gmt_rdt_data, gmt_rdt_emul, engine);
  if (m_doSys[GLT])
    ModifyCollection(glt_rdt_data, glt_rdt_emul, engine);
  if (m_doSys[GLT])
    ModifyCollection(glt_evm_data, glt_evm_emul, engine);
  if (m_doSys[GLT])
    ModifyCollection(glt_obj_data, glt_obj_emul, engine);

  if (verbose())
    std::cout << "L1EmulBias::produce - put...\n" << std::flush;

  /// append data into event
  if (m_doSys[ETP])
    iEvent.put(std::move(ecal_tp_data), instName[ETP][0]);
  if (m_doSys[HTP])
    iEvent.put(std::move(hcal_tp_data), instName[HTP][0]);
  if (m_doSys[RCT])
    iEvent.put(std::move(rct_em_data), instName[RCT][0]);
  if (m_doSys[RCT])
    iEvent.put(std::move(rct_rgn_data), instName[RCT][0]);
  if (m_doSys[GCT])
    iEvent.put(std::move(gct_isolaem_data), instName[GCT][0]);
  if (m_doSys[GCT])
    iEvent.put(std::move(gct_noisoem_data), instName[GCT][1]);
  if (m_doSys[GCT])
    iEvent.put(std::move(gct_cenjets_data), instName[GCT][2]);
  if (m_doSys[GCT])
    iEvent.put(std::move(gct_forjets_data), instName[GCT][3]);
  if (m_doSys[GCT])
    iEvent.put(std::move(gct_taujets_data), instName[GCT][4]);
  if (m_doSys[DTP])
    iEvent.put(std::move(dtp_ph_data), instName[DTP][0]);
  if (m_doSys[DTP])
    iEvent.put(std::move(dtp_th_data), instName[DTP][0]);
  if (m_doSys[DTF])
    iEvent.put(std::move(dtf_data), instName[DTF][0]);
  if (m_doSys[DTF])
    iEvent.put(std::move(dtf_trk_data), instName[DTF][1]);
  if (m_doSys[CTP])
    iEvent.put(std::move(ctp_data), instName[CTP][0]);
  if (m_doSys[CTF])
    iEvent.put(std::move(ctf_data), instName[CTF][0]);
  if (m_doSys[CTF])
    iEvent.put(std::move(ctf_trk_data), instName[CTF][1]);
  if (m_doSys[RPC])
    iEvent.put(std::move(rpc_cen_data), instName[RPC][0]);
  if (m_doSys[RPC])
    iEvent.put(std::move(rpc_for_data), instName[RPC][1]);
  if (m_doSys[LTC])
    iEvent.put(std::move(ltc_data), instName[LTC][0]);
  if (m_doSys[GMT])
    iEvent.put(std::move(gmt_data), instName[GMT][0]);
  if (m_doSys[GMT])
    iEvent.put(std::move(gmt_rdt_data), instName[GMT][0]);
  if (m_doSys[GLT])
    iEvent.put(std::move(glt_rdt_data), instName[GLT][0]);
  if (m_doSys[GLT])
    iEvent.put(std::move(glt_evm_data), instName[GLT][0]);
  if (m_doSys[GLT])
    iEvent.put(std::move(glt_obj_data), instName[GLT][0]);

  if (verbose())
    std::cout << "L1EmulBias::produce...done.\n" << std::flush;
}
