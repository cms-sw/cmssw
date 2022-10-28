#include "L1Trigger/HardwareValidation/plugins/L1DummyProducer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

using namespace dedefs;

L1DummyProducer::L1DummyProducer(const edm::ParameterSet& iConfig) {
  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag", 0);

  if (verbose())
    std::cout << "L1DummyProducer::L1DummyProducer...\n" << std::flush;

  std::vector<unsigned int> compColls = iConfig.getUntrackedParameter<std::vector<unsigned int> >("DO_SYSTEM");

  for (int i = 0; i < DEnsys; i++)
    m_doSys[i] = compColls[i];

  if (verbose()) {
    std::cout << "[L1DummyProducer] do sys? ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << m_doSys[i];
    std::cout << std::endl;
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

  if (verbose()) {
    std::cout << "[L1DummyProducer] instName:\n";
    for (int i = 0; i < DEnsys; i++)
      for (int j = 0; j < 5; j++)
        if (!instName[i][j].empty())
          std::cout << i << " " << SystLabel[i] << " " << j << " " << instName[i][j] << std::endl;
    std::cout << std::flush;
  }

  ///assertions/temporary
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

  /// list of collections to be produced
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

  ///rnd # settings
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "L1DummyProducer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }
  EBase_ = iConfig.getUntrackedParameter<double>("EnergyBase", 100.);
  ESigm_ = iConfig.getUntrackedParameter<double>("EnergySigm", 10.);

  nevt_ = 0;
}

L1DummyProducer::~L1DummyProducer() {}

void L1DummyProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());

  if (verbose())
    std::cout << "L1DummyProducer::produce...\n" << std::flush;

  /// define the data collections
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

  int nevt = nevt_++;
  /// fill candidate collections
  if (m_doSys[ETP])
    SimpleDigi(nevt, engine, ecal_tp_data);
  if (m_doSys[HTP])
    SimpleDigi(nevt, engine, hcal_tp_data);
  if (m_doSys[RCT])
    SimpleDigi(nevt, engine, rct_em_data);
  if (m_doSys[RCT])
    SimpleDigi(nevt, engine, rct_rgn_data);
  if (m_doSys[GCT])
    SimpleDigi(nevt, engine, gct_isolaem_data, 0);
  if (m_doSys[GCT])
    SimpleDigi(nevt, engine, gct_noisoem_data, 1);
  if (m_doSys[GCT])
    SimpleDigi(nevt, engine, gct_cenjets_data, 0);
  if (m_doSys[GCT])
    SimpleDigi(nevt, engine, gct_forjets_data, 1);
  if (m_doSys[GCT])
    SimpleDigi(nevt, engine, gct_taujets_data, 2);
  if (m_doSys[DTP])
    SimpleDigi(nevt, engine, dtp_ph_data);
  if (m_doSys[DTP])
    SimpleDigi(nevt, engine, dtp_th_data);
  if (m_doSys[DTF])
    SimpleDigi(nevt, engine, dtf_data, 0);
  if (m_doSys[DTF])
    SimpleDigi(nevt, engine, dtf_trk_data);
  if (m_doSys[CTP])
    SimpleDigi(nevt, engine, ctp_data);
  if (m_doSys[CTF])
    SimpleDigi(nevt, engine, ctf_data, 2);
  if (m_doSys[CTF])
    SimpleDigi(nevt, engine, ctf_trk_data);
  if (m_doSys[RPC])
    SimpleDigi(nevt, engine, rpc_cen_data, 1);
  if (m_doSys[RPC])
    SimpleDigi(nevt, engine, rpc_for_data, 3);
  if (m_doSys[LTC])
    SimpleDigi(nevt, engine, ltc_data);
  if (m_doSys[GMT])
    SimpleDigi(nevt, engine, gmt_data);
  if (m_doSys[GMT])
    SimpleDigi(nevt, engine, gmt_rdt_data);
  if (m_doSys[GLT])
    SimpleDigi(nevt, engine, glt_rdt_data);
  if (m_doSys[GLT])
    SimpleDigi(nevt, engine, glt_evm_data);
  if (m_doSys[GLT])
    SimpleDigi(nevt, engine, glt_obj_data);

  /// put collection
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
    std::cout << "L1DummyProducer::produce end.\n" << std::flush;
}
