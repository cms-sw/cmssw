#include "L1Trigger/Phase2L1GMT/interface/SAMuonCleaner.h"

SAMuonCleaner::SAMuonCleaner() {}

SAMuonCleaner::~SAMuonCleaner() {}

void SAMuonCleaner::overlapCleanTrack(l1t::SAMuon& source, const l1t::SAMuon& other, bool eq) {
  int rank1 = source.hwQual();
  int rank2 = other.hwQual();
  bool keep = true;
  if ((eq && rank1 <= rank2) || ((!eq) && rank1 < rank2))
    keep = false;
  l1t::MuonStubRefVector stubs;
  for (const auto& s1 : source.stubs()) {
    bool ok = true;
    for (const auto& s2 : other.stubs()) {
      if ((*s1) == (*s2) && (!keep))
        ok = false;
    }
    if (ok) {
      stubs.push_back(s1);
    }
  }
  source.setStubs(stubs);
}

void SAMuonCleaner::overlapCleanTrackInter(l1t::SAMuon& source, const l1t::SAMuon& other) {
  bool keep = false;
  l1t::MuonStubRefVector stubs;
  for (const auto& s1 : source.stubs()) {
    bool ok = true;
    for (const auto& s2 : other.stubs()) {
      if ((*s1) == (*s2) && (!keep))
        ok = false;
    }
    if (ok) {
      stubs.push_back(s1);
    }
  }
  source.setStubs(stubs);
}

std::vector<l1t::SAMuon> SAMuonCleaner::cleanTF(const std::vector<l1t::SAMuon>& tfMuons) {
  std::vector<l1t::SAMuon> out;
  for (unsigned int i = 0; i < tfMuons.size(); ++i) {
    l1t::SAMuon source = tfMuons[i];
    for (unsigned int j = 0; j < tfMuons.size(); ++j) {
      if (i == j)
        continue;
      overlapCleanTrack(source, tfMuons[j], false);
    }
    if (source.stubs().size() > 1)
      out.push_back(source);
  }
  return out;
}

std::vector<l1t::SAMuon> SAMuonCleaner::interTFClean(const std::vector<l1t::SAMuon>& bmtf,
                                                     const std::vector<l1t::SAMuon>& omtf,
                                                     const std::vector<l1t::SAMuon>& emtf) {
  std::vector<l1t::SAMuon> out = emtf;
  for (unsigned int i = 0; i < omtf.size(); ++i) {
    l1t::SAMuon source = omtf[i];
    for (const auto& other : emtf) {
      overlapCleanTrackInter(source, other);
    }
    if (source.stubs().size() > 1)
      out.push_back(source);
  }
  for (unsigned int i = 0; i < bmtf.size(); ++i) {
    l1t::SAMuon source = bmtf[i];
    for (const auto& other : omtf) {
      overlapCleanTrackInter(source, other);
    }
    if (source.stubs().size() > 1)
      out.push_back(source);
  }
  return out;
}

void SAMuonCleaner::swap(std::vector<l1t::SAMuon>& list, int i, int j) {
  const l1t::SAMuon& track1 = list[i];
  const l1t::SAMuon& track2 = list[j];

  int pt1 = track1.pt();
  int pt2 = track2.pt();
  bool swap = false;
  if (pt1 > pt2)
    swap = false;
  else
    swap = true;

  if (swap) {
    l1t::SAMuon tmp = list[i];
    list[i] = list[j];
    list[j] = tmp;
  }
}

void SAMuonCleaner::sort(std::vector<l1t::SAMuon>& in) {
  l1t::SAMuon nullTrack;
  nullTrack.setP4(reco::Candidate::LorentzVector(0, 0, 0, 0.000001));
  while (in.size() < 32)
    in.push_back(nullTrack);

  for (uint iteration = 0; iteration < 16; ++iteration) {
    for (uint i = 0; i < 32; i = i + 2) {
      swap(in, i, i + 1);
    }
    for (uint i = 1; i < 31; i = i + 2) {
      swap(in, i, i + 1);
    }
  }

  std::vector<l1t::SAMuon> out;
  for (const auto& track : in) {
    if (!track.stubs().empty() && out.size() < 12)
      out.push_back(track);
  }
  in = out;
}

std::vector<l1t::SAMuon> SAMuonCleaner::cleanTFMuons(const std::vector<l1t::SAMuon>& muons) {
  std::vector<l1t::SAMuon> out;

  //split into collections
  std::vector<l1t::SAMuon> bmtf;
  std::vector<l1t::SAMuon> omtf_pos;
  std::vector<l1t::SAMuon> emtf_pos;
  std::vector<l1t::SAMuon> omtf_neg;
  std::vector<l1t::SAMuon> emtf_neg;

  for (const auto& mu : muons) {
    if (mu.tfType() == l1t::tftype::bmtf) {
      bmtf.push_back(mu);
    } else if (mu.tfType() == l1t::tftype::omtf_pos) {
      omtf_pos.push_back(mu);
    } else if (mu.tfType() == l1t::tftype::emtf_pos) {
      emtf_pos.push_back(mu);
    } else if (mu.tfType() == l1t::tftype::omtf_neg) {
      omtf_neg.push_back(mu);
    } else if (mu.tfType() == l1t::tftype::emtf_neg) {
      emtf_neg.push_back(mu);
    }
  }

  std::vector<l1t::SAMuon> omtf_cleaned = cleanTF(omtf_pos);
  std::vector<l1t::SAMuon> omtf_neg_cleaned = cleanTF(omtf_neg);
  omtf_cleaned.insert(omtf_cleaned.end(), omtf_neg_cleaned.begin(), omtf_neg_cleaned.end());
  sort(omtf_cleaned);

  std::vector<l1t::SAMuon> emtf_cleaned = cleanTF(emtf_pos);
  std::vector<l1t::SAMuon> emtf_neg_cleaned = cleanTF(emtf_neg);
  emtf_cleaned.insert(emtf_cleaned.end(), emtf_neg_cleaned.begin(), emtf_neg_cleaned.end());
  sort(emtf_cleaned);

  std::vector<l1t::SAMuon> cleaned = interTFClean(bmtf, omtf_cleaned, emtf_cleaned);
  sort(cleaned);
  return cleaned;
}
