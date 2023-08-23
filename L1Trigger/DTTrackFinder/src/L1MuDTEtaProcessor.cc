//-------------------------------------------------
//
//   Class: L1MuDTEtaProcessor
//
//   Description: Eta Processor
//
//                An Eta Processor consists of :
//                a receiver unit,
//                one Eta Track Finder (ETF) and
//                one Eta Matching Unit (EMU)
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTEtaProcessor.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <bitset>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackSegEta.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPattern.h"
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

using namespace std;

// --------------------------------
//       class L1MuDTEtaProcessor
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTEtaProcessor::L1MuDTEtaProcessor(const L1MuDTTrackFinder& tf, int id, edm::ConsumesCollector iC)
    : m_tf(tf),
      m_epid(id),
      m_foundPattern(0),
      m_tseta(15),
      m_DTDigiToken(iC.consumes<L1MuDTChambThContainer>(m_tf.config()->getDTDigiInputTag())),
      theEtaToken(iC.esConsumes()),
      theQualToken(iC.esConsumes()),
      theMsksToken(iC.esConsumes()) {
  m_tseta.reserve(15);
}

//--------------
// Destructor --
//--------------

L1MuDTEtaProcessor::~L1MuDTEtaProcessor() {}

//--------------
// Operations --
//--------------

//
// run Eta Processor
//
void L1MuDTEtaProcessor::run(int bx, const edm::Event& e, const edm::EventSetup& c) {
  if (m_tf.config()->getEtaTF()) {
    receiveData(bx, e, c);
    runEtaTrackFinder(c);
  }

  receiveAddresses();
  runEtaMatchingUnit(c);

  assign();
}

//
// reset Eta Processor
//
void L1MuDTEtaProcessor::reset() {
  vector<const L1MuDTTrackSegEta*>::iterator iter = m_tseta.begin();
  while (iter != m_tseta.end()) {
    if (*iter) {
      delete *iter;
      *iter = nullptr;
    }
    iter++;
  }

  m_tseta.clear();

  for (int i = 0; i < 12; i++) {
    m_eta[i] = 99;
    m_fine[i] = false;
    m_pattern[i] = 0;
    m_address[i] = 0;
    m_TrackCand[i] = nullptr;
    m_TracKCand[i] = nullptr;
  }

  m_foundPattern.clear();

  m_mask = true;
}

//
// print track candidates found in Eta Processor
//
void L1MuDTEtaProcessor::print() const {
  bool empty1 = true;
  for (int i = 0; i < 15; i++) {
    empty1 &= (m_tseta[i] == nullptr || m_tseta[i]->empty());
  }

  bool empty2 = true;
  for (int i = 0; i < 12; i++) {
    empty2 &= (m_address[i] == 0);
  }

  if (!empty1 || !empty2) {
    cout << "Eta processor " << m_epid << " : " << endl;

    // print local pattern
    if (!empty1) {
      cout << "Local pattern : " << endl;
      for (int i = 0; i < 15; i++) {
        if ((i + 5) % 5 == 0)
          cout << "station " << m_tseta[i]->station() << " : ";
        bitset<7> pos(m_tseta[i]->position());
        bitset<7> qua(m_tseta[i]->quality());
        for (int j = 6; j >= 0; j--) {
          cout << pos[j] + qua[j];
        }
        cout << " ";
        if ((i + 1) % 5 == 0)
          cout << endl;
      }
      cout << "Found patterns :" << endl;
      vector<int>::const_iterator iter;
      for (iter = m_foundPattern.begin(); iter != m_foundPattern.end(); iter++) {
        const L1MuDTEtaPattern p = theEtaPatternLUT->getPattern(*iter);
        int qualitycode = p.quality();
        cout << "ID = " << setw(4) << p.id() << "  "
             << "eta = " << setw(3) << p.eta() << "  "
             << "quality = " << setw(2) << qualitycode << " (" << quality(qualitycode, 1) << " "
             << quality(qualitycode, 2) << " " << quality(qualitycode, 3) << ")";
        for (int i = 0; i < 12; i++) {
          if (m_pattern[i] == p.id())
            cout << " <--";
        }
        cout << endl;
      }
    }

    cout << "Received addresses : " << endl;
    for (int i = 0; i < 12; i++)
      cout << setw(3) << m_address[i] << " ";
    cout << endl;

    if (!empty1) {
      cout << "Matched patterns : " << endl;
      for (int i = 0; i < 12; i++) {
        if (m_fine[i]) {
          const L1MuDTEtaPattern p = theEtaPatternLUT->getPattern(m_pattern[i]);
          int fineeta = p.eta();
          int coarseeta = theQualPatternLUT->getCoarseEta(i / 2 + 1, m_address[i]);
          cout << "Index = " << setw(2) << i << ", "
               << "address = " << setw(2) << m_address[i] << " --> "
               << "pattern = " << setw(4) << m_pattern[i] << " "
               << "eta (coarse) = " << setw(3) << coarseeta << " "
               << "eta (fine) = " << setw(3) << fineeta << " "
               << "quality = " << setw(2) << p.quality() << endl;
        }
      }
    }

    cout << "Eta values and fine bits : " << endl;
    for (int i = 0; i < 12; i++)
      cout << setw(3) << m_eta[i] << " ";
    cout << endl;
    for (int i = 0; i < 12; i++)
      cout << setw(3) << m_fine[i] << " ";
    cout << endl;
  }
}

//
// receive data ( 15*3 DTBX eta trigger primitives )
//
void L1MuDTEtaProcessor::receiveData(int bx, const edm::Event& e, const edm::EventSetup& c) {
  msks = c.getHandle(theMsksToken);

  edm::Handle<L1MuDTChambThContainer> dttrig;
  e.getByToken(m_DTDigiToken, dttrig);

  // const int bx_offset = dttrig->correctBX();
  int bx_offset = 0;
  bx = bx + bx_offset;

  //
  // get 5*3 eta track segments
  //
  int sector = m_epid;
  for (int stat = 1; stat <= 3; stat++) {
    for (int wheel = -2; wheel <= 2; wheel++) {
      L1MuDTChambThDigi const* tseta = dttrig->chThetaSegm(wheel, stat, sector, bx);
      bitset<7> pos;
      bitset<7> qual;

      int lwheel = wheel + 1;
      if (wheel < 0)
        lwheel = wheel - 1;

      bool masked = false;
      if (stat == 1)
        masked = msks->get_etsoc_chdis_st1(lwheel, sector);
      if (stat == 2)
        masked = msks->get_etsoc_chdis_st2(lwheel, sector);
      if (stat == 3)
        masked = msks->get_etsoc_chdis_st3(lwheel, sector);

      if (!masked)
        m_mask = false;

      if (tseta && !masked) {
        if (wheel == -2 || wheel == -1 ||
            (wheel == 0 && (sector == 0 || sector == 3 || sector == 4 || sector == 7 || sector == 8 || sector == 11))) {
          for (int i = 0; i < 7; i++) {
            if (tseta->position(i))
              pos.set(6 - i);
            if (tseta->quality(i))
              qual.set(6 - i);
          }
        } else {
          for (int i = 0; i < 7; i++) {
            if (tseta->position(i))
              pos.set(i);
            if (tseta->quality(i))
              qual.set(i);
          }
        }
      }

      const L1MuDTTrackSegEta* tmpts =
          new L1MuDTTrackSegEta(wheel, sector, stat, pos.to_ulong(), qual.to_ulong(), bx - bx_offset);
      m_tseta.push_back(tmpts);
    }
  }
}

//
// receive track addresses from 6 Sector Processors
//
void L1MuDTEtaProcessor::receiveAddresses() {
  // get track address code of all track segments
  // 6*2 times 5 bits; valid range [1-22]

  int sector = m_epid;

  int i = 0;
  for (int wheel = -3; wheel <= 3; wheel++) {
    if (wheel == 0)
      continue;
    L1MuDTSecProcId tmpspid(wheel, sector);
    for (int number = 0; number < 2; number++) {
      const L1MuDTTrack* cand = m_tf.sp(tmpspid)->track(number);
      const L1MuDTTrack* canD = m_tf.sp(tmpspid)->tracK(number);
      if (cand) {
        m_address[i] = cand->address().trackAddressCode();
        if (!cand->empty()) {
          m_TrackCand[i] = const_cast<L1MuDTTrack*>(cand);
          m_TracKCand[i] = const_cast<L1MuDTTrack*>(canD);
        }
        i++;
      }
    }
  }
}

//
// run Eta Track Finder (ETF)
//
void L1MuDTEtaProcessor::runEtaTrackFinder(const edm::EventSetup& c) {
  theEtaPatternLUT = c.getHandle(theEtaToken);

  // check if there are any data
  bool empty = true;
  for (int i = 0; i < 15; i++) {
    empty &= m_tseta[i]->empty();
  }
  if (empty)
    return;

  // Pattern comparator:
  // loop over all patterns and compare with local chamber pattern
  // result : list of valid pattern IDs ( m_foundPattern )
  L1MuDTEtaPatternLut::ETFLut_iter it = theEtaPatternLUT->begin();
  while (it != theEtaPatternLUT->end()) {
    const L1MuDTEtaPattern pattern = (*it).second;
    int qualitycode = pattern.quality();

    bool good = true;

    for (int station = 0; station < 3; station++) {
      int q = quality(qualitycode, station + 1);
      int wheel = pattern.wheel(station + 1);
      int bin = pattern.position(station + 1);
      if (bin == 0)
        continue;
      bitset<7> pos = m_tseta[wheel + 2 + station * 5]->position();
      bitset<7> qual = m_tseta[wheel + 2 + station * 5]->quality();
      // compare position
      good &= pos.test(bin - 1);
      // compare quality
      if (q == 2)
        good &= qual.test(bin - 1);
    }

    if (good)
      m_foundPattern.push_back(pattern.id());

    it++;
  }
}

//
// run Eta Matching Unit (EMU)
//
void L1MuDTEtaProcessor::runEtaMatchingUnit(const edm::EventSetup& c) {
  theQualPatternLUT = c.getHandle(theQualToken);

  // loop over all addresses
  for (int i = 0; i < 12; i++) {
    int adr = m_address[i];
    if (adr == 0)
      continue;
    int sp = i / 2 + 1;  //sector processor [1,6]

    // assign coarse eta value
    if (!m_mask)
      m_eta[i] = theQualPatternLUT->getCoarseEta(sp, adr);
    if (m_eta[i] == 99)
      m_eta[i] = 32;
    if (m_eta[i] > 31)
      m_eta[i] -= 64;
    m_eta[i] += 32;

    if (m_foundPattern.empty())
      continue;

    // get list of qualified patterns ordered by quality
    // and compare with found patterns
    const vector<short>& qualifiedPatterns = theQualPatternLUT->getQualifiedPatterns(sp, adr);
    vector<short>::const_iterator iter;
    vector<int>::const_iterator f_iter;
    for (iter = qualifiedPatterns.begin(); iter != qualifiedPatterns.end(); iter++) {
      f_iter = find(m_foundPattern.begin(), m_foundPattern.end(), (*iter));
      // found
      if (f_iter != m_foundPattern.end()) {
        const L1MuDTEtaPattern p = theEtaPatternLUT->getPattern(*f_iter);
        // assign fine eta value
        m_fine[i] = true;
        m_eta[i] = p.eta();  // improved eta
        if (m_eta[i] == 99)
          m_eta[i] = 32;
        if (m_eta[i] > 31)
          m_eta[i] -= 64;
        m_eta[i] += 32;
        m_pattern[i] = (*f_iter);
        break;
      }
    }
  }

  // if both tracks from one sector processor deliver the same track address
  // both tracks get only a coarse eta value!

  // loop over sector processors
  for (int i = 0; i < 6; i++) {
    int idx1 = 2 * i;
    int idx2 = 2 * i + 1;
    int adr1 = m_address[idx1];
    int adr2 = m_address[idx2];
    if (adr1 == 0 || adr2 == 0)
      continue;
    if (adr1 == adr2 && !m_mask) {
      // both tracks get coarse (default) eta value
      m_eta[idx1] = theQualPatternLUT->getCoarseEta(i + 1, adr1);
      if (m_eta[idx1] == 99)
        m_eta[idx1] = 32;
      if (m_eta[idx1] > 31)
        m_eta[idx1] -= 64;
      m_eta[idx1] += 32;
      m_pattern[idx1] = 0;
      m_fine[idx1] = false;
      m_eta[idx2] = theQualPatternLUT->getCoarseEta(i + 1, adr2);
      if (m_eta[idx2] == 99)
        m_eta[idx2] = 32;
      if (m_eta[idx2] > 31)
        m_eta[idx2] -= 64;
      m_eta[idx2] += 32;
      m_pattern[idx2] = 0;
      m_fine[idx2] = false;
    }
  }
}

//
// assign values to track candidates
//
void L1MuDTEtaProcessor::assign() {
  for (int i = 0; i < 12; i++) {
    if (m_TrackCand[i]) {
      if (m_eta[i] != 99) {
        m_TrackCand[i]->setEta(m_eta[i]);
        m_TracKCand[i]->setEta(m_eta[i]);
      } else {
        //        if ( i/2 != 2 ) cerr << "L1MuDTEtaProcessor: assign invalid eta" << " " << m_address[i] << endl;
      }
      if (m_fine[i]) {
        m_TrackCand[i]->setFineEtaBit();
        m_TracKCand[i]->setFineEtaBit();
        // find all contributing track segments
        const L1MuDTEtaPattern p = theEtaPatternLUT->getPattern(m_pattern[i]);
        vector<const L1MuDTTrackSegEta*> TSeta;
        const L1MuDTTrackSegEta* ts = nullptr;
        for (int stat = 0; stat < 3; stat++) {
          int wh = p.wheel(stat + 1);
          int pos = p.position(stat + 1);
          if (pos == 0)
            continue;
          ts = m_tseta[wh + 2 + stat * 5];
          TSeta.push_back(ts);
        }
        m_TrackCand[i]->setTSeta(TSeta);
        m_TracKCand[i]->setTSeta(TSeta);
      }
    }
  }
}

//
// get quality:  id [0,26], stat [1,3]
//
int L1MuDTEtaProcessor::quality(int id, int stat) {
  // quality codes as defined in CMS Note 2001/027
  // This QualityPatterns are used to have a defined Quality-Identifier for
  // all possible found tracks.
  // Therefore three integer numbers ( Station 1, 2, 3 from left to right )
  // can have numbers like 0, 1 or 2
  //	0 ... no hit is given
  //	1 ... a LTRG is given
  //	2 ... a HTRG is given

  const int qualcode[27][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                               {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2},
                               {0, 2, 1}, {0, 1, 2}, {2, 2, 0}, {2, 0, 2}, {0, 2, 2}, {1, 1, 1}, {2, 1, 1},
                               {1, 2, 1}, {1, 1, 2}, {2, 2, 1}, {2, 1, 2}, {1, 2, 2}, {2, 2, 2}};

  return qualcode[id][stat - 1];
}
