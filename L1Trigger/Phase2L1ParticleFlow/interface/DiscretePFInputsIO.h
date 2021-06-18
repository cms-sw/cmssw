#ifndef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputsIO_H
#define L1Trigger_Phase2L1ParticleFlow_DiscretePFInputsIO_H

#include <cassert>
#include <cstdlib>
#include <cstdio>

#include "DiscretePFInputs.h"

namespace l1tpf_impl {
  inline void writeToFile(const CaloCluster &c, FILE *file) {
    fwrite(&c.hwPt, 2, 1, file);
    fwrite(&c.hwEmPt, 2, 1, file);
    fwrite(&c.hwPtErr, 2, 1, file);
    fwrite(&c.hwEta, 2, 1, file);
    fwrite(&c.hwPhi, 2, 1, file);
    fwrite(&c.hwFlags, 2, 1, file);
    fwrite(&c.isEM, 1, 1, file);
    // used is not written out
    // src is not written out
  }
  inline void readFromFile(CaloCluster &c, FILE *file) {
    fread(&c.hwPt, 2, 1, file);
    fread(&c.hwEmPt, 2, 1, file);
    fread(&c.hwPtErr, 2, 1, file);
    fread(&c.hwEta, 2, 1, file);
    fread(&c.hwPhi, 2, 1, file);
    fread(&c.hwFlags, 2, 1, file);
    fread(&c.isEM, 1, 1, file);
    c.used = false;
    c.src = nullptr;
  }

  inline void writeToFile(const InputTrack &t, FILE *file) {
    fwrite(&t.hwInvpt, 2, 1, file);
    fwrite(&t.hwVtxEta, 4, 1, file);
    fwrite(&t.hwVtxPhi, 4, 1, file);
    fwrite(&t.hwCharge, 1, 1, file);
    fwrite(&t.hwZ0, 2, 1, file);
    fwrite(&t.hwChi2, 2, 1, file);
    fwrite(&t.hwStubs, 2, 1, file);
    fwrite(&t.hwFlags, 2, 1, file);
    // src is not written out
  }
  inline void readFromFile(InputTrack &t, FILE *file) {
    fread(&t.hwInvpt, 2, 1, file);
    fread(&t.hwVtxEta, 4, 1, file);
    fread(&t.hwVtxPhi, 4, 1, file);
    fread(&t.hwCharge, 1, 1, file);
    fread(&t.hwZ0, 2, 1, file);
    fread(&t.hwChi2, 2, 1, file);
    fread(&t.hwStubs, 2, 1, file);
    fread(&t.hwFlags, 2, 1, file);
    t.src = nullptr;
  }
  inline void writeToFile(const PropagatedTrack &t, FILE *file) {
    writeToFile(static_cast<const InputTrack &>(t), file);
    fwrite(&t.hwPt, 2, 1, file);
    fwrite(&t.hwPtErr, 2, 1, file);
    fwrite(&t.hwCaloPtErr, 2, 1, file);
    fwrite(&t.hwEta, 2, 1, file);
    fwrite(&t.hwPhi, 2, 1, file);
    // muonLink, used, fromPV are transient
  }
  inline void readFromFile(PropagatedTrack &t, FILE *file) {
    readFromFile(static_cast<InputTrack &>(t), file);
    fread(&t.hwPt, 2, 1, file);
    fread(&t.hwPtErr, 2, 1, file);
    fread(&t.hwCaloPtErr, 2, 1, file);
    fread(&t.hwEta, 2, 1, file);
    fread(&t.hwPhi, 2, 1, file);
    t.muonLink = false;
    t.used = false;
    t.fromPV = false;
  }

  inline void writeToFile(const Muon &m, FILE *file) {
    fwrite(&m.hwPt, 2, 1, file);
    fwrite(&m.hwEta, 2, 1, file);
    fwrite(&m.hwPhi, 2, 1, file);
    fwrite(&m.hwFlags, 2, 1, file);
    fwrite(&m.hwCharge, 1, 1, file);
  }
  inline void readFromFile(Muon &m, FILE *file) {
    fread(&m.hwPt, 2, 1, file);
    fread(&m.hwEta, 2, 1, file);
    fread(&m.hwPhi, 2, 1, file);
    fread(&m.hwFlags, 2, 1, file);
    fread(&m.hwCharge, 1, 1, file);
    m.src = nullptr;
  }

  inline void writeToFile(const float &pug, FILE *file) { fwrite(&pug, sizeof(float), 1, file); }
  inline void readFromFile(float &pug, FILE *file) { fread(&pug, sizeof(float), 1, file); }

  template <typename T>
  void writeManyToFile(const std::vector<T> &objs, FILE *file) {
    uint32_t number = objs.size();
    fwrite(&number, 4, 1, file);
    for (uint32_t i = 0; i < number; ++i)
      writeToFile(objs[i], file);
  }

  template <typename T>
  void readManyFromFile(std::vector<T> &objs, FILE *file) {
    uint32_t number;
    fread(&number, 4, 1, file);
    objs.resize(number);
    for (uint32_t i = 0; i < number; ++i)
      readFromFile(objs[i], file);
  }

  inline void writeToFile(const InputRegion &r, FILE *file) {
    assert(4 == sizeof(float));
    fwrite(&r.etaCenter, 4, 1, file);
    fwrite(&r.etaMin, 4, 1, file);
    fwrite(&r.etaMax, 4, 1, file);
    fwrite(&r.phiCenter, 4, 1, file);
    fwrite(&r.phiHalfWidth, 4, 1, file);
    fwrite(&r.etaExtra, 4, 1, file);
    fwrite(&r.phiExtra, 4, 1, file);
    writeManyToFile(r.calo, file);
    writeManyToFile(r.emcalo, file);
    writeManyToFile(r.track, file);
    writeManyToFile(r.muon, file);
  }
  inline void readFromFile(InputRegion &r, FILE *file) {
    assert(4 == sizeof(float));
    fread(&r.etaCenter, 4, 1, file);
    fread(&r.etaMin, 4, 1, file);
    fread(&r.etaMax, 4, 1, file);
    fread(&r.phiCenter, 4, 1, file);
    fread(&r.phiHalfWidth, 4, 1, file);
    fread(&r.etaExtra, 4, 1, file);
    fread(&r.phiExtra, 4, 1, file);
    readManyFromFile(r.calo, file);
    readManyFromFile(r.emcalo, file);
    readManyFromFile(r.track, file);
    readManyFromFile(r.muon, file);
  }

}  // namespace l1tpf_impl
#endif
