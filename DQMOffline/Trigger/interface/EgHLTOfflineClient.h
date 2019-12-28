#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFLINECLIENT
#define DQMOFFLINE_TRIGGER_EGHLTOFFLINECLIENT

// -*- C++ -*-
//
// Package:    EgammaHLTOfflineClient
// Class:      EgammaHLTOffline
//
/*
 Description: This is a DQM client meant to plot high-level HLT trigger 
 quantities as stored in the HLT results object TriggerResults for the Egamma triggers

 Notes:
  Currently I would like to plot simple histograms of three seperate types of variables
  1) global event quantities: eg nr of electrons
  2) di-object quanities: transverse mass, di-electron mass
  3) single object kinematic and id variables: eg et,eta,isolation

*/
//
// Original Author:  Sam Harper
//         Created:  June 2008
//
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <vector>
#include <string>

class EgHLTOfflineClient : public DQMEDHarvester {
private:
  // DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string dirName_;

  std::vector<std::string> eleHLTFilterNames_;  //names of the filters monitored using electrons to make plots for
  std::vector<std::string> eleHLTFilterNames2Leg_;
  std::vector<std::string> eleTightLooseTrigNames_;
  std::vector<std::string> phoHLTFilterNames_;  //names of the filters monitored using photons to make plots for
  std::vector<std::string> phoHLTFilterNames2Leg_;
  std::vector<std::string> phoTightLooseTrigNames_;

  std::vector<std::string> eleN1EffVars_;
  std::vector<std::string> eleSingleEffVars_;
  std::vector<std::string> eleEffTags_;

  std::vector<std::string> phoN1EffVars_;
  std::vector<std::string> phoSingleEffVars_;
  std::vector<std::string> phoEffTags_;

  std::vector<std::string> eleTrigTPEffVsVars_;
  std::vector<std::string> phoTrigTPEffVsVars_;
  std::vector<std::string> eleLooseTightTrigEffVsVars_;
  std::vector<std::string> phoLooseTightTrigEffVsVars_;

  std::vector<std::string> eleHLTvOfflineVars_;
  std::vector<std::string> phoHLTvOfflineVars_;

  bool runClientEndLumiBlock_;
  bool runClientEndRun_;
  bool runClientEndJob_;

  bool filterInactiveTriggers_;
  bool isSetup_;
  std::string hltTag_;

public:
  explicit EgHLTOfflineClient(const edm::ParameterSet&);
  ~EgHLTOfflineClient() override;

  // virtual void beginJob();
  // virtual void analyze(const edm::Event&, const edm::EventSetup&); //dummy
  // virtual void endJob();
  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  // virtual void endRun(const edm::Run& run, const edm::EventSetup& c);

  // virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& context){}
  // DQM Client Diagnostic
  // virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;  //performed in the endJob
  void dqmEndLuminosityBlock(DQMStore::IBooker&,
                             DQMStore::IGetter&,
                             edm::LuminosityBlock const&,
                             edm::EventSetup const&) override;  //performed in the endLumi

  //at somepoint these all may migrate to a helper class
  void createN1EffHists(const std::string& filterName,
                        const std::string& baseName,
                        const std::string& region,
                        const std::vector<std::string>& varNames,
                        DQMStore::IBooker&,
                        DQMStore::IGetter&);

  void createSingleEffHists(const std::string& filterName,
                            const std::string& baseName,
                            const std::string& region,
                            const std::vector<std::string>& varNames,
                            DQMStore::IBooker&,
                            DQMStore::IGetter&);

  void createLooseTightTrigEff(const std::vector<std::string>& tightLooseTrigNames,
                               const std::string& region,
                               const std::vector<std::string>& vsVarNames,
                               const std::string& objName,
                               DQMStore::IBooker&,
                               DQMStore::IGetter&);

  void createTrigTagProbeEffHists(const std::string& filterName,
                                  const std::string& region,
                                  const std::vector<std::string>& vsVarNames,
                                  const std::string& objName,
                                  DQMStore::IBooker&,
                                  DQMStore::IGetter&);

  void createTrigTagProbeEffHistsNewAlgo(const std::string& filterName,
                                         const std::string& region,
                                         const std::vector<std::string>& vsVarNames,
                                         const std::string& objName,
                                         DQMStore::IBooker&,
                                         DQMStore::IGetter&);

  void createTrigTagProbeEffHists2Leg(const std::string& filterNameLeg1,
                                      const std::string& filterNameLeg2,
                                      const std::string& region,
                                      const std::vector<std::string>& vsVarNames,
                                      const std::string& objName,
                                      DQMStore::IBooker&,
                                      DQMStore::IGetter&);

  void createHLTvsOfflineHists(const std::string& filterName,
                               const std::string& baseName,
                               const std::string& region,
                               const std::vector<std::string>& varNames,
                               DQMStore::IBooker&,
                               DQMStore::IGetter&);

  MonitorElement* FillHLTvsOfflineHist(const std::string& filter,
                                       const std::string& name,
                                       const std::string& title,
                                       const MonitorElement* numer,
                                       const MonitorElement* denom,
                                       DQMStore::IBooker&,
                                       DQMStore::IGetter&);

  MonitorElement* makeEffMonElemFromPassAndAll(const std::string& filterName,
                                               const std::string& name,
                                               const std::string& title,
                                               const MonitorElement* pass,
                                               const MonitorElement* all,
                                               DQMStore::IBooker&,
                                               DQMStore::IGetter&);

  MonitorElement* makeEffMonElemFromPassAndFail(const std::string& filterName,
                                                const std::string& name,
                                                const std::string& title,
                                                const MonitorElement* pass,
                                                const MonitorElement* fail,
                                                DQMStore::IBooker&,
                                                DQMStore::IGetter&);

  MonitorElement* makeEffMonElemFromPassAndFailAndTagTag(const std::string& filter,
                                                         const std::string& name,
                                                         const std::string& title,
                                                         const MonitorElement* pass,
                                                         const MonitorElement* fail,
                                                         const MonitorElement* tagtag,
                                                         DQMStore::IBooker&,
                                                         DQMStore::IGetter&);

  MonitorElement* makeEffMonElem2Leg(const std::string& filter,
                                     const std::string& name,
                                     const std::string& title,
                                     const MonitorElement* Leg1Eff,
                                     const MonitorElement* Leg2NotLeg1Source,
                                     const MonitorElement* all,
                                     DQMStore::IBooker&,
                                     DQMStore::IGetter&);

private:
  void runClient_(DQMStore::IBooker&, DQMStore::IGetter&);  //master function which runs the client
};

#endif
