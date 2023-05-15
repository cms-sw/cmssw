//-------------------------------------------------
//
//   Class: DTTSPhi.cpp
//
//   Description: Implementation of TS Phi trigger algorithm
//
//
//   Author List:
//   C. Grandi
//   Modifications:
//   jan02 - D.Bonacorsi/S.Marcellini
//           improved algorithm for 2nd track handling in case of pile-up in TSM
//           param: tsmgetcarryflag - value: 1 (default)
//   feb04 - Implementation of sector collector related stuff(S. Marcellini)
//   jan07 - C. Battilana local conf update
//   mar07 - S. Vanini : parameters from DTConfigManager
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSPhi.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTTraco/interface/DTTracoCard.h"
#include "L1Trigger/DTTraco/interface/DTTracoTrigData.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSCand.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSM.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSS.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//----------------
DTTSPhi::DTTSPhi(DTTrigGeom *geom, DTTracoCard *tracocard) : DTGeomSupplier(geom), _tracocard(tracocard) {
  // reserve the appropriate amount of space for vectors
  int i = 0;
  for (i = 0; i < DTConfigTSPhi::NSTEPL - DTConfigTSPhi::NSTEPF + 1; i++) {  // SM add + 1
    _tss[i].reserve(DTConfigTSPhi::NTSSTSM);
    // DBSM-doubleTSM
    _tsm[i].reserve(DTConfigTSPhi::NTSMD);
  }

  for (int is = 0; is < DTConfigTSPhi::NSTEPL - DTConfigTSPhi::NSTEPF + 1; is++) {
    // create DTTSSs
    for (int itss = 1; itss <= DTConfigTSPhi::NTSSTSM; itss++) {
      DTTSS *tss = new DTTSS(itss);
      _tss[is].push_back(tss);
    }

    // create DTTSMs     SM double TSM
    for (int itsmd = 1; itsmd <= DTConfigTSPhi::NTSMD; itsmd++) {
      DTTSM *tsm = new DTTSM(itsmd);
      _tsm[is].push_back(tsm);
    }
  }
}

//--------------
// Destructor --
//--------------
DTTSPhi::~DTTSPhi() {
  std::vector<DTTSS *>::iterator ptss;
  std::vector<DTTSM *>::iterator ptsm;
  for (int is = 0; is < DTConfigTSPhi::NSTEPL - DTConfigTSPhi::NSTEPF + 1; is++) {
    // clear TSSs
    for (ptss = _tss[is].begin(); ptss != _tss[is].end(); ptss++) {
      delete (*ptss);
    }
    _tss[is].clear();
    // clear TSMs
    for (ptsm = _tsm[is].begin(); ptsm != _tsm[is].end(); ptsm++) {
      delete (*ptsm);
    }
    _tsm[is].clear();
  }

  localClear();

  // delete _config;
}

//--------------
// Operations --
//--------------

void DTTSPhi::localClear() {
  for (int is = 0; is < DTConfigTSPhi::NSTEPL - DTConfigTSPhi::NSTEPF + 1; is++) {
    // clear buffer
    std::vector<DTTSCand *>::iterator p1;
    for (p1 = _tctrig[is].begin(); p1 != _tctrig[is].end(); p1++) {
      delete (*p1);
    }
    _tctrig[is].clear();

    std::vector<DTTSS *>::iterator ptss;
    for (ptss = _tss[is].begin(); ptss != _tss[is].end(); ptss++) {
      (*ptss)->clear();
    }
    // clear all DTTSM
    std::vector<DTTSM *>::iterator ptsm;
    for (ptsm = _tsm[is].begin(); ptsm != _tsm[is].end(); ptsm++) {
      (*ptsm)->clear();
    }
  }
}

void DTTSPhi::setConfig(const DTConfigManager *conf) {
  DTChamberId sid = ChamberId();
  _config = conf->getDTConfigTSPhi(sid);

  for (int is = 0; is < DTConfigTSPhi::NSTEPL - DTConfigTSPhi::NSTEPF + 1; is++) {
    // set TSS config
    std::vector<DTTSS *>::iterator ptss;
    for (ptss = _tss[is].begin(); ptss != _tss[is].end(); ptss++) {
      (*ptss)->setConfig(config());
    }
    // set TSM config
    std::vector<DTTSM *>::iterator ptsm;
    for (ptsm = _tsm[is].begin(); ptsm != _tsm[is].end(); ptsm++) {
      (*ptsm)->setConfig(config());
    }
  }
}

void DTTSPhi::loadTSPhi() {
  // clear DTTSSs and DTTSM
  localClear();

  if (config()->debug()) {
    edm::LogInfo("DTTSPhi") << "loadDTTSPhi called for wheel=" << wheel() << ", station=" << station()
                            << ", sector=" << sector();
  }

  // loop on all TRACO triggers
  std::vector<DTTracoTrigData>::const_iterator p;
  std::vector<DTTracoTrigData>::const_iterator pend = _tracocard->end();
  for (p = _tracocard->begin(); p != pend; p++) {
    if (config()->usedTraco(p->tracoNumber()) /*|| config()->usedTraco(p->tracoNumber())==1*/) {
      int step = p->step();
      int fs = (p->isFirst()) ? 1 : 2;

      // if first track is found inhibit second track processing in previous BX
      if (fs == 1 && step > DTConfigTSPhi::NSTEPF)
        ignoreSecondTrack(step - 1, p->tracoNumber());

      // load trigger
      addTracoT(step, &(*p), fs);
    }
  }
}

void DTTSPhi::addTracoT(int step, const DTTracoTrigData *tracotrig, int ifs) {
  if (step < DTConfigTSPhi::NSTEPF || step > DTConfigTSPhi::NSTEPL) {
    edm::LogWarning("DTTSPhi") << "addTracoT: step out of range: " << step << " trigger not added!";
    return;
  }
  // Check that a preview is present and code is not zero
  if (!tracotrig->pvCode() || !tracotrig->code()) {
    edm::LogWarning("DTTSPhi") << "addTracoT: preview not present in TRACO trigger or its code=0 "
                               << " trigger not added!";
    return;
  }

  // Get the appropriate TSS
  int itss = (tracotrig->tracoNumber() - 1) / DTConfigTSPhi::NTCTSS + 1;
  if (itss < 1 || itss > DTConfigTSPhi::NTSSTSM) {
    edm::LogWarning("DTTSPhi") << "addTracoT: wrong TRACO number: " << tracotrig->tracoNumber()
                               << " trigger not added!";
    return;
  }

  // TSM status check (if it is the case, reject TRACO triggers related to
  // broken TSMData)
  if (config()->TsmStatus().element(itss) == 0) {  // TSMD broken
    return;
  }

  int pos = tracotrig->tracoNumber() - (itss - 1) * DTConfigTSPhi::NTCTSS;
  DTTSS *tss = getDTTSS(step, itss);

  // Create a new Trigger Server candidate
  DTTSCand *cand = new DTTSCand(tss, tracotrig, ifs, pos);

  // Add it to the buffer and to the TSS
  _tctrig[step - DTConfigTSPhi::NSTEPF].push_back(cand);
  tss->addDTTSCand(cand);

  // Debugging...
  if (config()->debug()) {
    edm::LogInfo("DTTSPhi") << "addTracoT at step " << step;
    if (ifs == 1) {
      edm::LogWarning("DTTSPhi") << " (first track)";
    } else {
      edm::LogWarning("DTTSPhi") << " (second track)";
    }
    edm::LogWarning("DTTSPhi") << " from TRACO " << tracotrig->tracoNumber() << " to TSS " << tss->number()
                               << ", position=" << pos;
    tracotrig->print();
  }
  // end debugging
}

void DTTSPhi::runTSPhi() {
  DTTSCand *secondPrevBx = nullptr;  // new DTTSCand;

  bool existSecondPrevBx = false;
  int itsmd = 1;  // initialize it to 1, default value if not in back up mode
  int ntsm[DTConfigTSPhi::NSTEPL + 1 - DTConfigTSPhi::NSTEPF][DTConfigTSPhi::NTSMD];
  int i_tsmd;

  for (int is = DTConfigTSPhi::NSTEPF; is < DTConfigTSPhi::NSTEPL + 1; is++) {
    // loop on DTTSSs
    i_tsmd = 0;
    ntsm[is - DTConfigTSPhi::NSTEPF][0] = 0;  // counter to make sector collector run if at least a tsm
    ntsm[is - DTConfigTSPhi::NSTEPF][1] = 0;
    std::vector<DTTSS *>::iterator p;
    for (p = _tss[is - DTConfigTSPhi::NSTEPF].begin(); p < _tss[is - DTConfigTSPhi::NSTEPF].end(); p++) {
      if ((*p)->nTracoT(1) > 0) {
        // run DTTSS algorithm on non-empty DTTSSs
        (*p)->run();
        // load DTTSM with output DTTSS tracks
        if ((*p)->nTracks() > 0) {
          for (int it = 1; it <= (*p)->nTracks(); it++) {
            //--- SM double TSM    get the corresponding tsm data
            int bkmod = config()->TsmStatus().element(0);
            if (bkmod == 0) {                // we are in back-up mode
              int my_itss = (*p)->number();  // metodo di DTTSS che ritorna itss
              int ntsstsmd = config()->TSSinTSMD(station(), sector());
              if (ntsstsmd < 2 || ntsstsmd > DTConfigTSPhi::NTSSTSMD) {
                edm::LogWarning("DTTSPhi") << " addTracoT - wrong TSMD: " << ntsstsmd;
              }

              // Get the appropriate TSMD
              itsmd = (my_itss - 1) / ntsstsmd + 1;
              if (config()->debug()) {
                edm::LogInfo("DTTSPhi") << " addTracoT: itsmd = (my_itss -1 ) / ntsstsmd + 1  ---> my_itss = "
                                        << my_itss << "  ntsstsmd = " << ntsstsmd << "  itsmd = " << itsmd;
              }
            } else if (bkmod == 1) {
              itsmd = 1;  // initialize it to 1, default value if not in back up mode
            }
            if (itsmd > 2)
              edm::LogWarning("DTTSPhi") << "****** RunTSPhi wrong  itsmd = " << itsmd;
            DTTSM *tsm = getDTTSM(is, itsmd);
            tsm->addCand((*p)->getTrack(it));
          }
        }  // end loop on output DTTSS tracks
      }
    }  // end loop on DTTSSs

    // at least a DTTSS with signal. Run DTTSM

    std::vector<DTTSM *>::iterator p_tsm;

    for (p_tsm = _tsm[is - DTConfigTSPhi::NSTEPF].begin(); p_tsm < _tsm[is - DTConfigTSPhi::NSTEPF].end(); p_tsm++) {
      // Run TSM sorting if at least a first track

      i_tsmd = (*p_tsm)->number() - 1;  // returns itsmd (0 in default, 0 or 1 when bkmode )

      if ((*p_tsm)->nCand(1) > 0) {
        int bkmod = config()->TsmStatus().element(0);

        (*p_tsm)->run(bkmod);  // bkmod 1 normal, 0 backup
        // Run TSM for current BX in case of 1st Tracks
        // Run TSM for previous BX for second tracks, to check whether there is
        // a pile up Tells whether a second track at previous BX exists

        if ((*p_tsm)->nTracks() > 0) {
          // We have a first track. Store it if code is > 0

          if ((*p_tsm)->getTrack(1)->tracoTr()->code() > 0) {
            DTTSCand *first = (*p_tsm)->getTrack(1);
            if (config()->TsmGetCarryFlag() == 0) {  //  get 1st tk at current BX and ignore any 2nd tk at
                                                     //  previous BX

              _cache.push_back(DTChambPhSegm(ChamberId(), is, (*p_tsm)->getTrack(1)->tracoTr(), 1));
              ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;  // SM increment ntsm at current BX
              if (config()->debug())
                edm::LogInfo("DTTSPhi") << "ntsm = " << ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd] << " is = " << is
                                        << " i_tsmd = " << i_tsmd;
              if ((*p_tsm)->nTracks() > 1) {                         // there is a 2nd tk
                if ((*p_tsm)->getTrack(2)->tracoTr()->code() > 0) {  // check if its code > 0
                  ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;
                  if (config()->debug())
                    edm::LogInfo("DTTSPhi") << "ntsm = " << ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd] << " is = " << is
                                            << " i_tsmd = " << i_tsmd;

                  secondPrevBx = (*p_tsm)->getTrack(2);  // assign second tk of previous BX
                }
              }
            } else if (config()->TsmGetCarryFlag() == 1) {  // compare with 2nd tk in previous BX and get the tk
                                                            // with better quality
              existSecondPrevBx =
                  ((is - 1 - DTConfigTSPhi::NSTEPF >= 0) && (ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd] > 1) &&
                   (secondPrevBx->tracoTr()->code() > 0));
              if ((!existSecondPrevBx) ||
                  !((secondPrevBx->isCorr() && secondPrevBx->isHtrig() && secondPrevBx->isInner()) ||
                    (secondPrevBx->isCorr() && secondPrevBx->isHtrig() && !secondPrevBx->isInner()) ||
                    (!secondPrevBx->isCorr() && secondPrevBx->isHtrig() && secondPrevBx->isInner())) ||

                  ((secondPrevBx->isCorr() && secondPrevBx->isHtrig() && secondPrevBx->isInner()) &&
                   (first->isCorr() && first->isHtrig() && first->isInner())) ||

                  ((secondPrevBx->isCorr() && secondPrevBx->isHtrig() && !secondPrevBx->isInner()) &&
                   ((first->isCorr() && first->isHtrig() && first->isInner()) ||
                    (first->isCorr() && first->isHtrig() && !first->isInner()))) ||

                  ((!secondPrevBx->isCorr() && secondPrevBx->isHtrig() && secondPrevBx->isInner()) &&
                   !((!first->isCorr() && first->isHtrig() && !first->isInner()) ||
                     (!first->isCorr() && !first->isHtrig() && first->isInner()) ||
                     (!first->isCorr() && !first->isHtrig() && !first->isInner()) ||
                     (first->isCorr() && !first->isHtrig() && !first->isInner()) ||
                     (first->isCorr() && !first->isHtrig() && first->isInner())))) {
                // SM sector collector
                ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;  // SM increment ntsm at current BX. I need to
                                                             // know if there is at least a first track from
                                                             // TSM to run Sect Coll

                _cache.push_back(DTChambPhSegm(ChamberId(), is, (*p_tsm)->getTrack(1)->tracoTr(), 1));
                //		(*p_tsm)->getTrack(1)->print();

                if ((*p_tsm)->nTracks() > 1) {  // there is a 2nd tk
                  ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;
                  if ((*p_tsm)->getTrack(2)->tracoTr()->code() > 0) {  // check if its code > 0
                    secondPrevBx = (*p_tsm)->getTrack(2);              // assign second previous BX
                  }
                }
              } else {  // if 2nd tk prev BX is better than first present BX skip
                        // the event and get 2nd prev BX
                ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd]++;  // SM increment ntsm at previous BX.
                _cache.push_back(DTChambPhSegm(ChamberId(), is - 1, secondPrevBx->tracoTr(), 2));
                // secondPrevBx->print();
              }
            }

            else if (config()->TsmGetCarryFlag() == 2) {  // neglect first tk if it is a low uncorrelated
                                                          // trigger
              existSecondPrevBx =
                  ((is - 1 - DTConfigTSPhi::NSTEPF >= 0) && (ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd] > 1) &&
                   (secondPrevBx->tracoTr()->code() > 0));
              if ((!existSecondPrevBx) || first->isHtrig() || first->isCorr()) {
                ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;  // SM increment ntsm at current BX.
                // SM sector collector: Load DTSectColl with output of DTTSM
                _cache.push_back(DTChambPhSegm(ChamberId(), is, (*p_tsm)->getTrack(1)->tracoTr(), 1));
                //		(*p_tsm)->getTrack(1)->print();

                if ((*p_tsm)->nTracks() > 1) {  // there is a 2nd tk
                  ntsm[is - DTConfigTSPhi::NSTEPF][i_tsmd]++;
                  if ((*p_tsm)->getTrack(2)->tracoTr()->code() > 0) {  // check if its code > 0
                    secondPrevBx = (*p_tsm)->getTrack(2);              // assign second tk of previous BX
                  }
                }
              } else {
                ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd]++;  // SM increment ntsm at previous BX.
                _cache.push_back(DTChambPhSegm(ChamberId(), is - 1, secondPrevBx->tracoTr(), 2));
                //		secondPrevBx->print();
              }
            }
          }
        }

      } else if (((*p_tsm)->nCand(1) == 0) && (is - 1 - DTConfigTSPhi::NSTEPF >= 0) &&
                 ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd] >
                     0) {  // it means that the last BX with sort 2 was not the
                           // previous one
        existSecondPrevBx =
            ((is - 1 - DTConfigTSPhi::NSTEPF >= 0) && (ntsm[is - 1 - DTConfigTSPhi::NSTEPF][i_tsmd] > 1) &&
             (secondPrevBx->tracoTr()->code() > 0));
        if (existSecondPrevBx) {
          _cache.push_back(DTChambPhSegm(ChamberId(), is - 1, secondPrevBx->tracoTr(), 2));

          //	  secondPrevBx->print();
        }
      }
    }
    //---

  }  // end loop on step
  // debugging...
  if (config()->debug()) {
    if (!_cache.empty()) {
      edm::LogInfo("DTTSPhi") << " Phi segments ";
      std::vector<DTChambPhSegm>::const_iterator p;
      for (p = _cache.begin(); p < _cache.end(); p++) {
        p->print();
      }
    }
  }
  //  end debugging
}

void DTTSPhi::ignoreSecondTrack(int step, int tracon) {
  int itsmd = 1;  // initialize it to default

  if (step < DTConfigTSPhi::NSTEPF || step > DTConfigTSPhi::NSTEPL) {
    edm::LogWarning("DTTSPhi") << "ignoreSecondTrack: step out of range: " << step << " no flag set!";
    return;
  }
  int itss = (tracon - 1) / DTConfigTSPhi::NTCTSS + 1;
  if (itss < 1 || itss > DTConfigTSPhi::NTSSTSM) {
    edm::LogWarning("DTTSPhi") << "ignoreSecondTrack: wrong TRACO number: " << tracon << " no flag set!";
    return;
  }
  DTTSS *tss = getDTTSS(step, itss);
  tss->ignoreSecondTrack();

  int bkmod = config()->TsmStatus().element(0);
  if (bkmod == 0) {  // we are in back-up mode

    int ntsstsmd = config()->TSSinTSMD(station(), sector());
    // Get the appropriate TSMD
    itsmd = (itss - 1) / ntsstsmd + 1;
  }

  DTTSM *tsm = getDTTSM(step, itsmd);
  tsm->ignoreSecondTrack();
}

DTTSS *DTTSPhi::getDTTSS(int step, unsigned n) const {
  if (step < DTConfigTSPhi::NSTEPF || step > DTConfigTSPhi::NSTEPL) {
    edm::LogWarning("DTTSPhi") << "getDTTSS: step out of range: " << step << " empty pointer returned!";
    return nullptr;
  }
  if (n < 1 || n > _tss[step - DTConfigTSPhi::NSTEPF].size()) {
    edm::LogWarning("DTTSPhi") << "getDTTSS: requested DTTSS not present: " << n << " (at step " << step << ")"
                               << " empty pointer returned!";
    return nullptr;
  }

  std::vector<DTTSS *>::const_iterator p = _tss[step - DTConfigTSPhi::NSTEPF].begin() + n - 1;
  return *p;
}

DTTSM *DTTSPhi::getDTTSM(int step, unsigned n) const {
  if (step < DTConfigTSPhi::NSTEPF || step > DTConfigTSPhi::NSTEPL) {
    edm::LogWarning("DTTSPhi") << "getDTTSM: step out of range: " << step << " empty pointer returned!";
    return nullptr;
  }
  if (n < 1 || n > _tsm[step - DTConfigTSPhi::NSTEPF].size()) {
    edm::LogWarning("DTTSPhi") << "getDTTSM: requested DTTSM not present: " << n << " (at step " << step << ")"
                               << " empty pointer returned!";
    return nullptr;
  }
  std::vector<DTTSM *>::const_iterator p_tsm = _tsm[step - DTConfigTSPhi::NSTEPF].begin() + n - 1;
  return *p_tsm;
}

int DTTSPhi::nSegm(int step) {
  int n = 0;
  std::vector<DTChambPhSegm>::const_iterator p;  // p=0;
  for (p = begin(); p < end(); p++) {
    if (p->step() == step)
      n++;
  }
  return n;
}

const DTChambPhSegm *DTTSPhi::segment(int step, unsigned n) {
  std::vector<DTChambPhSegm>::const_iterator p;  // p=0;
  for (p = begin(); p < end(); p++) {
    if (p->step() == step && ((n == 1 && p->isFirst()) || (n == 2 && !p->isFirst())))
      return &(*p);  // p;
  }
  return nullptr;
}

LocalPoint DTTSPhi::localPosition(const DTTrigData *tr) const {
  //@@ patch for Sun 4.2 compiler
  // sm DTChambPhSegm* trig =
  // dynamic_cast<DTChambPhSegm*>(const_cast<DTTrigData*>(tr));
  const DTChambPhSegm *trig = dynamic_cast<const DTChambPhSegm *>(tr);
  if (!trig) {
    edm::LogWarning("DTTSPhi") << "LocalPosition called with wrong argument!";
    return LocalPoint(0, 0, 0);
  }
  return _tracocard->localPosition(trig->tracoTrig());
}

LocalVector DTTSPhi::localDirection(const DTTrigData *tr) const {
  DTChambPhSegm *trig = dynamic_cast<DTChambPhSegm *>(const_cast<DTTrigData *>(tr));
  //  const DTChambPhSegm* trig = dynamic_cast<const DTChambPhSegm*>(tr);
  if (!trig) {
    edm::LogWarning("DTTSPhi") << "LocalDirection called with wrong argument!";
    return LocalVector(0, 0, 0);
  }
  return _tracocard->localDirection(trig->tracoTrig());
}
