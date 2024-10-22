/*
 * \file L1TGMT.cc
 *
 * \author J. Berryhill, I. Mikulec
 *
 */

#include "DQM/L1TMonitor/interface/L1TGMT.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

using namespace std;
using namespace edm;

const double L1TGMT::piconv_ = 180. / acos(-1.);

L1TGMT::L1TGMT(const ParameterSet& ps)
    : verbose_(ps.getUntrackedParameter<bool>("verbose", false))  // verbosity switch
      ,
      gmtSource_(consumes<L1MuGMTReadoutCollection>(ps.getParameter<InputTag>("gmtSource"))),
      bxnum_old_(0),
      obnum_old_(0),
      trsrc_old_(0) {
  if (verbose_)
    cout << "L1TGMT: constructor...." << endl;
  l1muTrigscaleToken_ = esConsumes<edm::Transition::BeginRun>();
  l1TrigptscaleToken_ = esConsumes<edm::Transition::BeginRun>();
}

L1TGMT::~L1TGMT() {}

void L1TGMT::analyze(const Event& e, const EventSetup& c) {
  if (verbose_)
    cout << "L1TGMT: analyze...." << endl;

  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  e.getByToken(gmtSource_, pCollection);

  if (!pCollection.isValid()) {
    edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection";
    return;
  }

  // remember the bx of 1st candidate of each system (9=none)
  int bx1st[4] = {9, 9, 9, 9};

  // get GMT readout collection
  L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
  // get record vector
  vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  // loop over records of individual bx's
  vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

  for (RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++) {
    vector<L1MuRegionalCand> INPCands[4] = {
        RRItr->getDTBXCands(), RRItr->getBrlRPCCands(), RRItr->getCSCCands(), RRItr->getFwdRPCCands()};
    vector<L1MuGMTExtendedCand> GMTCands = RRItr->getGMTCands();

    vector<L1MuRegionalCand>::const_iterator INPItr;
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
    vector<L1MuGMTExtendedCand>::const_iterator GMTItr2;

    int BxInEvent = RRItr->getBxInEvent();

    // count non-empty candidates in this bx
    int nSUBS[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
      for (INPItr = INPCands[i].begin(); INPItr != INPCands[i].end(); ++INPItr) {
        if (!INPItr->empty()) {
          nSUBS[i]++;
          if (bx1st[i] == 9)
            bx1st[i] = BxInEvent;
        }
      }
      subs_nbx[i]->Fill(float(nSUBS[i]), float(BxInEvent));
    }

    for (GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr) {
      if (!GMTItr->empty())
        nSUBS[GMT]++;
    }
    subs_nbx[GMT]->Fill(float(nSUBS[GMT]), float(BxInEvent));

    ////////////////////////////////////////////////////////////////////////////////////////////
    // from here care only about the L1A bunch crossing
    if (BxInEvent != 0)
      continue;

    // get the absolute bx number of the L1A
    int Bx = RRItr->getBxNr();

    bx_number->Fill(double(Bx));

    for (int i = 0; i < 4; i++) {
      for (INPItr = INPCands[i].begin(); INPItr != INPCands[i].end(); ++INPItr) {
        if (INPItr->empty())
          continue;
        subs_eta[i]->Fill(INPItr->etaValue());
        subs_phi[i]->Fill(phiconv_(INPItr->phiValue()));
        subs_pt[i]->Fill(INPItr->ptValue());
        subs_qty[i]->Fill(INPItr->quality());
        subs_etaphi[i]->Fill(INPItr->etaValue(), phiconv_(INPItr->phiValue()));
        subs_etaqty[i]->Fill(INPItr->etaValue(), INPItr->quality());
        int word = INPItr->getDataWord();
        for (int j = 0; j < 32; j++) {
          if (word & (1 << j))
            subs_bits[i]->Fill(float(j));
        }
      }
    }

    for (GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr) {
      if (GMTItr->empty())
        continue;
      subs_eta[GMT]->Fill(GMTItr->etaValue());
      subs_phi[GMT]->Fill(phiconv_(GMTItr->phiValue()));
      subs_pt[GMT]->Fill(GMTItr->ptValue());
      subs_qty[GMT]->Fill(GMTItr->quality());
      subs_etaphi[GMT]->Fill(GMTItr->etaValue(), phiconv_(GMTItr->phiValue()));
      subs_etaqty[GMT]->Fill(GMTItr->etaValue(), GMTItr->quality());
      int word = GMTItr->getDataWord();
      for (int j = 0; j < 32; j++) {
        if (word & (1 << j))
          subs_bits[GMT]->Fill(float(j));
      }

      if (GMTItr->isMatchedCand()) {
        if (GMTItr->quality() > 3) {
          eta_dtcsc_and_rpc->Fill(GMTItr->etaValue());
          phi_dtcsc_and_rpc->Fill(phiconv_(GMTItr->phiValue()));
          etaphi_dtcsc_and_rpc->Fill(GMTItr->etaValue(), phiconv_(GMTItr->phiValue()));
        }
      } else if (GMTItr->isRPC()) {
        if (GMTItr->quality() > 3) {
          eta_rpc_only->Fill(GMTItr->etaValue());
          phi_rpc_only->Fill(phiconv_(GMTItr->phiValue()));
          etaphi_rpc_only->Fill(GMTItr->etaValue(), phiconv_(GMTItr->phiValue()));
        }
      } else {
        if (GMTItr->quality() > 3) {
          eta_dtcsc_only->Fill(GMTItr->etaValue());
          phi_dtcsc_only->Fill(phiconv_(GMTItr->phiValue()));
          etaphi_dtcsc_only->Fill(GMTItr->etaValue(), phiconv_(GMTItr->phiValue()));
        }

        if (GMTItr != GMTCands.end()) {
          for (GMTItr2 = GMTCands.begin(); GMTItr2 != GMTCands.end(); ++GMTItr2) {
            if (GMTItr2 == GMTItr)
              continue;
            if (GMTItr2->empty())
              continue;
            if (GMTItr2->isRPC()) {
              if (GMTItr->isFwd()) {
                dist_eta_csc_rpc->Fill(GMTItr->etaValue() - GMTItr2->etaValue());
                dist_phi_csc_rpc->Fill(phiconv_(GMTItr->phiValue()) - phiconv_(GMTItr2->phiValue()));
              } else {
                dist_eta_dt_rpc->Fill(GMTItr->etaValue() - GMTItr2->etaValue());
                dist_phi_dt_rpc->Fill(phiconv_(GMTItr->phiValue()) - phiconv_(GMTItr2->phiValue()));
              }
            } else {
              if (!(GMTItr->isFwd()) && GMTItr2->isFwd()) {
                dist_eta_dt_csc->Fill(GMTItr->etaValue() - GMTItr2->etaValue());
                dist_phi_dt_csc->Fill(phiconv_(GMTItr->phiValue()) - phiconv_(GMTItr2->phiValue()));
              } else if (GMTItr->isFwd() && !(GMTItr2->isFwd())) {
                dist_eta_dt_csc->Fill(GMTItr2->etaValue() - GMTItr->etaValue());
                dist_phi_dt_csc->Fill(phiconv_(GMTItr->phiValue()) - phiconv_(GMTItr2->phiValue()));
              }
            }
          }
        }
      }
    }

    n_rpcb_vs_dttf->Fill(float(nSUBS[DTTF]), float(nSUBS[RPCb]));
    n_rpcf_vs_csctf->Fill(float(nSUBS[CSCTF]), float(nSUBS[RPCf]));
    n_csctf_vs_dttf->Fill(float(nSUBS[DTTF]), float(nSUBS[CSCTF]));

    regional_triggers->Fill(-1.);  // fill underflow for normalization
    if (nSUBS[GMT])
      regional_triggers->Fill(0.);  // fill all muon bin
    int ioff = 1;
    for (int i = 0; i < 4; i++) {
      if (nSUBS[i])
        regional_triggers->Fill(float(5 * i + nSUBS[i] + ioff));
    }
    if (nSUBS[DTTF] && (nSUBS[RPCb] || nSUBS[RPCf]))
      regional_triggers->Fill(22.);
    if (nSUBS[DTTF] && nSUBS[CSCTF])
      regional_triggers->Fill(23.);
    if (nSUBS[CSCTF] && (nSUBS[RPCb] || nSUBS[RPCf]))
      regional_triggers->Fill(24.);
    if (nSUBS[DTTF] && nSUBS[CSCTF] && (nSUBS[RPCb] || nSUBS[RPCf]))
      regional_triggers->Fill(25.);

    // fill only if previous event corresponds to previous trigger
    //    if( (Ev - evnum_old_) == 1 && bxnum_old_ > -1 ) {
    // assume getting all events in a sequence (usefull only from reco data)
    if (bxnum_old_ > -1) {
      float dBx = Bx - bxnum_old_ + 3564.0 * (e.orbitNumber() - obnum_old_);
      for (int id = 0; id < 4; id++) {
        if (trsrc_old_ & (1 << id)) {
          for (int i = 0; i < 4; i++) {
            if (nSUBS[i])
              subs_dbx[i]->Fill(dBx, float(id));
          }
        }
      }
    }

    // save quantities for the next event
    bxnum_old_ = Bx;
    obnum_old_ = e.orbitNumber();
    trsrc_old_ = 0;
    for (int i = 0; i < 4; i++) {
      if (nSUBS[i])
        trsrc_old_ |= (1 << i);
    }
  }

  if (bx1st[DTTF] < 9 && bx1st[RPCb] < 9)
    bx_dt_rpc->Fill(bx1st[DTTF], bx1st[RPCb]);
  if (bx1st[CSCTF] < 9 && bx1st[RPCf] < 9)
    bx_csc_rpc->Fill(bx1st[CSCTF], bx1st[RPCf]);
  if (bx1st[DTTF] < 9 && bx1st[CSCTF] < 9)
    bx_dt_csc->Fill(bx1st[DTTF], bx1st[CSCTF]);
}

double L1TGMT::phiconv_(float phi) {
  double phiout = double(phi);
  phiout *= piconv_;
  phiout += 0.001;  // add a small value to get off the bin edge
  return phiout;
}

void L1TGMT::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& c) {
  std::string subs[5] = {"DTTF", "RPCb", "CSCTF", "RPCf", "GMT"};

  const L1MuTriggerScales* scales = &c.getData(l1muTrigscaleToken_);
  const L1MuTriggerPtScale* scalept = &c.getData(l1TrigptscaleToken_);

  ibooker.setCurrentFolder("L1T/L1TGMT");

  int nqty = 8;
  double qtymin = -0.5;
  double qtymax = 7.5;

  float phiscale[145];
  int nphiscale;
  {
    int nbins = scales->getPhiScale()->getNBins();
    if (nbins > 144)
      nbins = 144;
    for (int j = 0; j <= nbins; j++) {
      phiscale[j] = piconv_ * scales->getPhiScale()->getValue(j);
    }
    nphiscale = nbins;
  }

  float qscale[9];
  {
    for (int j = 0; j < 9; j++) {
      qscale[j] = -0.5 + j;
    }
  }

  // pt scale first bin reserved for empty muon
  float ptscale[32];
  int nptscale;
  {
    int nbins = scalept->getPtScale()->getNBins() - 1;
    if (nbins > 31)
      nbins = 31;
    for (int j = 1; j <= nbins; j++) {
      ptscale[j - 1] = scalept->getPtScale()->getValue(j);
    }
    ptscale[nbins] = ptscale[nbins - 1] + 10.;  // make reasonable size last bin
    nptscale = nbins;
  }

  float etascale[5][66];
  int netascale[5];
  // DTTF eta scale
  {
    int nbins = scales->getRegionalEtaScale(DTTF)->getNBins();
    if (nbins > 65)
      nbins = 65;
    for (int j = 0; j <= nbins; j++) {
      etascale[DTTF][j] = scales->getRegionalEtaScale(DTTF)->getValue(j);
    }
    netascale[DTTF] = nbins;
  }
  // RPCb etascale
  {
    int nbins = scales->getRegionalEtaScale(RPCb)->getNBins();
    if (nbins > 65)
      nbins = 65;
    for (int j = 0; j <= nbins; j++) {
      etascale[RPCb][j] = scales->getRegionalEtaScale(RPCb)->getValue(j);
    }
    netascale[RPCb] = nbins;
  }
  // CSCTF etascale
  // special case - need to mirror 2*32 bins
  {
    int nbins = scales->getRegionalEtaScale(CSCTF)->getNBins();
    if (nbins > 32)
      nbins = 32;

    int i = 0;
    for (int j = nbins; j >= 0; j--, i++) {
      etascale[CSCTF][i] = (-1) * scales->getRegionalEtaScale(CSCTF)->getValue(j);
    }
    for (int j = 0; j <= nbins; j++, i++) {
      etascale[CSCTF][i] = scales->getRegionalEtaScale(CSCTF)->getValue(j);
    }
    netascale[CSCTF] = i - 1;
  }
  // RPCf etascale
  {
    int nbins = scales->getRegionalEtaScale(RPCf)->getNBins();
    if (nbins > 65)
      nbins = 65;
    for (int j = 0; j <= nbins; j++) {
      etascale[RPCf][j] = scales->getRegionalEtaScale(RPCf)->getValue(j);
    }
    netascale[RPCf] = nbins;
  }
  // GMT etascale
  {
    int nbins = scales->getGMTEtaScale()->getNBins();
    if (nbins > 32)
      nbins = 32;

    int i = 0;
    for (int j = nbins; j > 0; j--, i++) {
      etascale[GMT][i] = (-1) * scales->getGMTEtaScale()->getValue(j);
    }
    for (int j = 0; j <= nbins; j++, i++) {
      etascale[GMT][i] = scales->getGMTEtaScale()->getValue(j);
    }
    netascale[GMT] = i - 1;
  }

  std::string hname("");
  std::string htitle("");

  for (int i = 0; i < 5; i++) {
    hname = subs[i] + "_nbx";
    htitle = subs[i] + " multiplicity in bx";
    subs_nbx[i] = ibooker.book2D(hname.data(), htitle.data(), 4, 1., 5., 5, -2.5, 2.5);
    subs_nbx[i]->setAxisTitle(subs[i] + " candidates", 1);
    subs_nbx[i]->setAxisTitle("bx wrt L1A", 2);

    hname = subs[i] + "_eta";
    htitle = subs[i] + " eta value";
    subs_eta[i] = ibooker.book1D(hname.data(), htitle.data(), netascale[i], etascale[i]);
    subs_eta[i]->setAxisTitle("eta", 1);

    hname = subs[i] + "_phi";
    htitle = subs[i] + " phi value";
    subs_phi[i] = ibooker.book1D(hname.data(), htitle.data(), nphiscale, phiscale);
    subs_phi[i]->setAxisTitle("phi (deg)", 1);

    hname = subs[i] + "_pt";
    htitle = subs[i] + " pt value";
    subs_pt[i] = ibooker.book1D(hname.data(), htitle.data(), nptscale, ptscale);
    subs_pt[i]->setAxisTitle("L1 pT (GeV)", 1);

    hname = subs[i] + "_qty";
    htitle = subs[i] + " qty value";
    subs_qty[i] = ibooker.book1D(hname.data(), htitle.data(), nqty, qtymin, qtymax);
    subs_qty[i]->setAxisTitle(subs[i] + " quality", 1);

    hname = subs[i] + "_etaphi";
    htitle = subs[i] + " phi vs eta";
    subs_etaphi[i] = ibooker.book2D(hname.data(), htitle.data(), netascale[i], etascale[i], nphiscale, phiscale);
    subs_etaphi[i]->setAxisTitle("eta", 1);
    subs_etaphi[i]->setAxisTitle("phi (deg)", 2);

    hname = subs[i] + "_etaqty";
    htitle = subs[i] + " qty vs eta";
    subs_etaqty[i] = ibooker.book2D(hname.data(), htitle.data(), netascale[i], etascale[i], nqty, qscale);
    subs_etaqty[i]->setAxisTitle("eta", 1);
    subs_etaqty[i]->setAxisTitle(subs[i] + " quality", 2);

    hname = subs[i] + "_bits";
    htitle = subs[i] + " bit population";
    subs_bits[i] = ibooker.book1D(hname.data(), htitle.data(), 32, -0.5, 31.5);
    subs_bits[i]->setAxisTitle("bit number", 1);
  }

  regional_triggers = ibooker.book1D("Regional_trigger", "Muon trigger contribution", 27, 0., 27.);
  regional_triggers->setAxisTitle("regional trigger", 1);
  int ib = 1;
  regional_triggers->setBinLabel(ib++, "All muons", 1);
  ib++;
  regional_triggers->setBinLabel(ib++, "DT 1mu", 1);
  regional_triggers->setBinLabel(ib++, "DT 2mu", 1);
  regional_triggers->setBinLabel(ib++, "DT 3mu", 1);
  regional_triggers->setBinLabel(ib++, "DT 4mu", 1);
  ib++;
  regional_triggers->setBinLabel(ib++, "RPCb 1mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCb 2mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCb 3mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCb 4mu", 1);
  ib++;
  regional_triggers->setBinLabel(ib++, "CSC 1mu", 1);
  regional_triggers->setBinLabel(ib++, "CSC 2mu", 1);
  regional_triggers->setBinLabel(ib++, "CSC 3mu", 1);
  regional_triggers->setBinLabel(ib++, "CSC 4mu", 1);
  ib++;
  regional_triggers->setBinLabel(ib++, "RPCf 1mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCf 2mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCf 3mu", 1);
  regional_triggers->setBinLabel(ib++, "RPCf 4mu", 1);
  ib++;
  regional_triggers->setBinLabel(ib++, "DT & RPC", 1);
  regional_triggers->setBinLabel(ib++, "DT & CSC", 1);
  regional_triggers->setBinLabel(ib++, "CSC & RPC", 1);
  regional_triggers->setBinLabel(ib++, "DT & CSC & RPC", 1);

  bx_number = ibooker.book1D("Bx_Number", "Bx number ROP chip", 3564, 0., 3564.);
  bx_number->setAxisTitle("bx number", 1);

  dbx_chip = ibooker.bookProfile("dbx_Chip", "bx count difference wrt ROP chip", 5, 0., 5., 100, -4000., 4000., "i");
  dbx_chip->setAxisTitle("chip name", 1);
  dbx_chip->setAxisTitle("delta bx", 2);
  dbx_chip->setBinLabel(1, "IND", 1);
  dbx_chip->setBinLabel(2, "INB", 1);
  dbx_chip->setBinLabel(3, "INC", 1);
  dbx_chip->setBinLabel(4, "INF", 1);
  dbx_chip->setBinLabel(5, "SRT", 1);

  eta_dtcsc_and_rpc =
      ibooker.book1D("eta_DTCSC_and_RPC", "eta of confirmed GMT candidates", netascale[GMT], etascale[GMT]);
  eta_dtcsc_and_rpc->setAxisTitle("eta", 1);

  eta_dtcsc_only =
      ibooker.book1D("eta_DTCSC_only", "eta of unconfirmed DT/CSC candidates", netascale[GMT], etascale[GMT]);
  eta_dtcsc_only->setAxisTitle("eta", 1);

  eta_rpc_only = ibooker.book1D("eta_RPC_only", "eta of unconfirmed RPC candidates", netascale[GMT], etascale[GMT]);
  eta_rpc_only->setAxisTitle("eta", 1);

  phi_dtcsc_and_rpc = ibooker.book1D("phi_DTCSC_and_RPC", "phi of confirmed GMT candidates", nphiscale, phiscale);
  phi_dtcsc_and_rpc->setAxisTitle("phi (deg)", 1);

  phi_dtcsc_only = ibooker.book1D("phi_DTCSC_only", "phi of unconfirmed DT/CSC candidates", nphiscale, phiscale);
  phi_dtcsc_only->setAxisTitle("phi (deg)", 1);

  phi_rpc_only = ibooker.book1D("phi_RPC_only", "phi of unconfirmed RPC candidates", nphiscale, phiscale);
  phi_rpc_only->setAxisTitle("phi (deg)", 1);

  etaphi_dtcsc_and_rpc = ibooker.book2D("etaphi_DTCSC_and_RPC",
                                        "eta vs phi map of confirmed GMT candidates",
                                        netascale[GMT],
                                        etascale[GMT],
                                        nphiscale,
                                        phiscale);
  etaphi_dtcsc_and_rpc->setAxisTitle("eta", 1);
  etaphi_dtcsc_and_rpc->setAxisTitle("phi (deg)", 2);

  etaphi_dtcsc_only = ibooker.book2D("etaphi_DTCSC_only",
                                     "eta vs phi map of unconfirmed DT/CSC candidates",
                                     netascale[GMT],
                                     etascale[GMT],
                                     nphiscale,
                                     phiscale);
  etaphi_dtcsc_only->setAxisTitle("eta", 1);
  etaphi_dtcsc_only->setAxisTitle("phi (deg)", 2);

  etaphi_rpc_only = ibooker.book2D("etaphi_RPC_only",
                                   "eta vs phi map of unconfirmed RPC candidates",
                                   netascale[GMT],
                                   etascale[GMT],
                                   nphiscale,
                                   phiscale);
  etaphi_rpc_only->setAxisTitle("eta", 1);
  etaphi_rpc_only->setAxisTitle("phi (deg)", 2);

  dist_phi_dt_rpc = ibooker.book1D("dist_phi_DT_RPC", "Dphi between DT and RPC candidates", 100, -125., 125.);
  dist_phi_dt_rpc->setAxisTitle("delta phi (deg)", 1);

  dist_phi_csc_rpc = ibooker.book1D("dist_phi_CSC_RPC", "Dphi between CSC and RPC candidates", 100, -125., 125.);
  dist_phi_csc_rpc->setAxisTitle("delta phi (deg)", 1);

  dist_phi_dt_csc = ibooker.book1D("dist_phi_DT_CSC", "Dphi between DT and CSC candidates", 100, -125., 125.);
  dist_phi_dt_csc->setAxisTitle("delta phi (deg)", 1);

  dist_eta_dt_rpc = ibooker.book1D("dist_eta_DT_RPC", "Deta between DT and RPC candidates", 40, -1., 1.);
  dist_eta_dt_rpc->setAxisTitle("delta eta", 1);

  dist_eta_csc_rpc = ibooker.book1D("dist_eta_CSC_RPC", "Deta between CSC and RPC candidates", 40, -1., 1.);
  dist_eta_csc_rpc->setAxisTitle("delta eta", 1);

  dist_eta_dt_csc = ibooker.book1D("dist_eta_DT_CSC", "Deta between DT and CSC candidates", 40, -1., 1.);
  dist_eta_dt_csc->setAxisTitle("delta eta", 1);

  n_rpcb_vs_dttf = ibooker.book2D("n_RPCb_vs_DTTF", "n cands RPCb vs DTTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
  n_rpcb_vs_dttf->setAxisTitle("DTTF candidates", 1);
  n_rpcb_vs_dttf->setAxisTitle("barrel RPC candidates", 2);

  n_rpcf_vs_csctf = ibooker.book2D("n_RPCf_vs_CSCTF", "n cands RPCf vs CSCTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
  n_rpcf_vs_csctf->setAxisTitle("CSCTF candidates", 1);
  n_rpcf_vs_csctf->setAxisTitle("endcap RPC candidates", 2);

  n_csctf_vs_dttf = ibooker.book2D("n_CSCTF_vs_DTTF", "n cands CSCTF vs DTTF", 5, -0.5, 4.5, 5, -0.5, 4.5);
  n_csctf_vs_dttf->setAxisTitle("DTTF candidates", 1);
  n_csctf_vs_dttf->setAxisTitle("CSCTF candidates", 2);

  bx_dt_rpc = ibooker.book2D("bx_DT_vs_RPC", "1st bx DT vs. RPC", 5, -2.5, 2.5, 5, -2.5, 2.5);
  bx_dt_rpc->setAxisTitle("bx of 1st DTTF candidate", 1);
  bx_dt_rpc->setAxisTitle("bx of 1st RPCb candidate", 2);

  bx_csc_rpc = ibooker.book2D("bx_CSC_vs_RPC", "1st bx CSC vs. RPC", 5, -2.5, 2.5, 5, -2.5, 2.5);
  bx_csc_rpc->setAxisTitle("bx of 1st CSCTF candidate", 1);
  bx_csc_rpc->setAxisTitle("bx of 1st RPCf candidate", 2);

  bx_dt_csc = ibooker.book2D("bx_DT_vs_CSC", "1st bx DT vs. CSC", 5, -2.5, 2.5, 5, -2.5, 2.5);
  bx_dt_csc->setAxisTitle("bx of 1st DTTF candidate", 1);
  bx_dt_csc->setAxisTitle("bx of 1st CSCTF candidate", 2);

  for (int i = 0; i < 4; i++) {
    hname = subs[i] + "_dbx";
    htitle = "dBx " + subs[i] + " to previous event";
    subs_dbx[i] = ibooker.book2D(hname.data(), htitle.data(), 1000, 0., 1000., 4, 0., 4.);
    for (int j = 0; j < 4; j++) {
      subs_dbx[i]->setBinLabel((j + 1), subs[j], 2);
    }
  }
}
