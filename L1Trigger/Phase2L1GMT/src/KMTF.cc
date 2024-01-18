#include "L1Trigger/Phase2L1GMT/interface/KMTF.h"
using namespace Phase2L1GMT;

KMTF::KMTF(int verbose, const edm::ParameterSet& iConfig) : verbose_(verbose), trackMaker_(new KMTFCore(iConfig)) {}

KMTF::~KMTF() {}

std::pair<std::vector<l1t::KMTFTrack>, std::vector<l1t::KMTFTrack> > KMTF::process(
    const l1t::MuonStubRefVector& stubsAll, int bx, unsigned int MAXN) {
  std::vector<l1t::KMTFTrack> pretracksP2;
  std::vector<l1t::KMTFTrack> pretracksP3;
  std::vector<l1t::KMTFTrack> pretracksP4;
  std::vector<l1t::KMTFTrack> pretracksD2;
  std::vector<l1t::KMTFTrack> pretracksD3;
  std::vector<l1t::KMTFTrack> pretracksD4;
  uint Nstubs4 = 0;
  uint Nstubs3 = 0;
  uint Nstubs2 = 0;
  uint Nstubs1 = 0;

  l1t::MuonStubRefVector stubs4;
  l1t::MuonStubRefVector stubs3;
  l1t::MuonStubRefVector stubs2;
  l1t::MuonStubRefVector stubs1;

  for (const auto& stub : stubsAll) {
    if (stub->bxNum() != bx || stub->id() > 3)
      continue;
    if (!stub->isBarrel())
      continue;

    if (stub->depthRegion() == 4) {
      if (Nstubs4 < MAXN) {
        stubs4.push_back(stub);
        Nstubs4++;
      }
    }
    if (stub->depthRegion() == 3) {
      if (Nstubs3 < MAXN) {
        stubs3.push_back(stub);
        Nstubs3++;
      }
    }
    if (stub->depthRegion() == 2) {
      if (Nstubs2 < MAXN) {
        stubs2.push_back(stub);
        Nstubs2++;
      }
    }
    if (stub->depthRegion() == 1) {
      if (Nstubs1 < MAXN) {
        stubs1.push_back(stub);
        Nstubs1++;
      }
    }
  }

  //Sort the seeds by tag so that the emulator is aligned like the firmware
  SeedSorter sorter;
  if (stubs4.size() > 1) {
    std::sort(stubs4.begin(), stubs4.end(), sorter);
  }
  if (stubs3.size() > 1) {
    std::sort(stubs3.begin(), stubs3.end(), sorter);
  }
  if (stubs2.size() > 1) {
    std::sort(stubs2.begin(), stubs2.end(), sorter);
  }
  if (stubs1.size() > 1) {
    std::sort(stubs1.begin(), stubs1.end(), sorter);
  }

  bool pre_patterns = (verbose_ > 1) && ((Nstubs4 + Nstubs3 + Nstubs2 + Nstubs1) > 2);

  //OK now process the data almost as in hardware
  for (unsigned int i = 0; i < 32; ++i) {
    //print the stubs taking into account

    bool patterns = pre_patterns && ((i < Nstubs4) || (i < Nstubs3) || (i << Nstubs2));

    if (patterns) {
      printf("KMTFPattern ");
      if (i < Nstubs4)
        printf("%d %d %d 1 %d 0 ",
               stubs4[0]->coord1(),
               stubs4[0]->coord2(),
               stubs4[0]->quality(),
               stubs4[0]->kmtf_address());
      else
        printf("0 0 0 0 511 0 ");

      if (i < Nstubs3) {
        for (const auto& s : stubs3) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
        //pad with zeros
        for (unsigned int j = stubs3.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
      } else {
        for (unsigned int j = stubs3.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
        for (const auto& s : stubs3) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
      }

      if (i < Nstubs2) {
        for (const auto& s : stubs2) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
        //pad with zeros
        for (unsigned int j = stubs2.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
      } else {
        for (unsigned int j = stubs2.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
        for (const auto& s : stubs2) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
      }
      if (i < Nstubs1) {
        for (const auto& s : stubs1) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
        //pad with zeros
        for (unsigned int j = stubs1.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
      } else {
        for (unsigned int j = stubs1.size(); j < 32; ++j) {
          printf("0 0 0 0 511 0 ");
        }
        for (const auto& s : stubs1) {
          printf("%d %d %d 1 %d 0 ", s->coord1(), s->coord2(), s->quality(), s->kmtf_address());
        }
      }
    }

    //seed is 4
    if (i < Nstubs4) {
      l1t::MuonStubRefVector stubs_proc;
      if (Nstubs3 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs3.begin(), stubs3.end());
      if (Nstubs2 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs2.begin(), stubs2.end());
      if (Nstubs1 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs1.begin(), stubs1.end());
      std::pair<l1t::KMTFTrack, l1t::KMTFTrack> tracks = trackMaker_->chain(stubs4[0], stubs_proc);
      if (tracks.first.id() & 0x1)
        pretracksP4.push_back(tracks.first);
      if (tracks.second.id() & 0x2)
        pretracksD4.push_back(tracks.second);
      if (patterns) {
        if (tracks.first.id() & 0x1)
          printf("1 %d %d %d %d %d %d ",
                 tracks.first.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.first.ptPrompt(),
                 tracks.first.phiAtMuon() / (1 << 5),
                 tracks.first.coarseEta(),
                 int(tracks.first.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.first.rankPrompt());
        else
          printf("0 0 0 0 0 0 0 ");
        if (tracks.second.id() & 0x2)
          printf("1 %d %d %d %d %d %d ",
                 tracks.second.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.second.ptDisplaced(),
                 tracks.second.phiAtMuon() / (1 << 5),
                 tracks.second.coarseEta(),
                 int(tracks.second.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.second.rankDisp());
        else
          printf("0 0 0 0 0 0 0 ");
      }
    } else if (patterns) {
      printf("0 0 0 0 0 0 0 ");
      printf("0 0 0 0 0 0 0 ");
    }

    if (i < Nstubs3) {
      l1t::MuonStubRefVector stubs_proc;
      if (Nstubs2 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs2.begin(), stubs2.end());
      if (Nstubs1 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs1.begin(), stubs1.end());
      std::pair<l1t::KMTFTrack, l1t::KMTFTrack> tracks = trackMaker_->chain(stubs3[0], stubs_proc);
      if (tracks.first.id() & 0x1)
        pretracksP3.push_back(tracks.first);
      if (tracks.second.id() & 0x2)
        pretracksD3.push_back(tracks.second);
      if (patterns) {
        if (tracks.first.id() & 0x1)
          printf("1 %d %d %d %d %d %d ",
                 tracks.first.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.first.ptPrompt(),
                 tracks.first.phiAtMuon() / (1 << 5),
                 tracks.first.coarseEta(),
                 int(tracks.first.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.first.rankPrompt());
        else
          printf("0 0 0 0 0 0 0 ");
        if (tracks.second.id() & 0x2)
          printf("1 %d %d %d %d %d %d ",
                 tracks.second.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.second.ptDisplaced(),
                 tracks.second.phiAtMuon() / (1 << 5),
                 tracks.second.coarseEta(),
                 int(tracks.second.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.second.rankDisp());
        else
          printf("0 0 0 0 0 0 0 ");
      }
    } else if (patterns) {
      printf("0 0 0 0 0 0 0 ");
      printf("0 0 0 0 0 0 0 ");
    }
    if (i < Nstubs2) {
      l1t::MuonStubRefVector stubs_proc;
      if (Nstubs1 > 0)
        stubs_proc.insert(stubs_proc.end(), stubs1.begin(), stubs1.end());
      std::pair<l1t::KMTFTrack, l1t::KMTFTrack> tracks = trackMaker_->chain(stubs2[0], stubs_proc);
      if (tracks.first.id() & 0x1)
        pretracksP2.push_back(tracks.first);
      if (tracks.second.id() & 0x2)
        pretracksD2.push_back(tracks.second);
      if (patterns) {
        if (tracks.first.id() & 0x1)
          printf("1 %d %d %d %d %d %d ",
                 tracks.first.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.first.ptPrompt(),
                 tracks.first.phiAtMuon() / (1 << 5),
                 tracks.first.coarseEta(),
                 int(tracks.first.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.first.rankPrompt());
        else
          printf("0 0 0 0 0 0 0 ");
        if (tracks.second.id() & 0x2)
          printf("1 %d %d %d %d %d %d\n",
                 tracks.second.curvatureAtVertex() < 0 ? 1 : 0,
                 tracks.second.ptDisplaced(),
                 tracks.second.phiAtMuon() / (1 << 5),
                 tracks.second.coarseEta(),
                 int(tracks.second.dxy() * ap_ufixed<8, 1>(1.606)),
                 tracks.second.rankDisp());
        else
          printf("0 0 0 0 0 0 0\n");
      }
    } else if (patterns) {
      printf("0 0 0 0 0 0 0 ");
      printf("0 0 0 0 0 0 0\n");
    }
    //Now the shift register emulation in C_++
    if (stubs4.size() > 1) {
      l1t::MuonStubRef s4 = stubs4[0];
      stubs4.erase(stubs4.begin(), stubs4.begin() + 1);
      stubs4.push_back(s4);
    }
    if (stubs3.size() > 1) {
      l1t::MuonStubRef s3 = stubs3[0];
      stubs3.erase(stubs3.begin(), stubs3.begin() + 1);
      stubs3.push_back(s3);
    }
    if (stubs2.size() > 1) {
      l1t::MuonStubRef s2 = stubs2[0];
      stubs2.erase(stubs2.begin(), stubs2.begin() + 1);
      stubs2.push_back(s2);
    }
    if (stubs1.size() > 1) {
      l1t::MuonStubRef s1 = stubs1[0];
      stubs1.erase(stubs1.begin(), stubs1.begin() + 1);
      stubs1.push_back(s1);
    }
  }

  std::vector<l1t::KMTFTrack> cleanedPrompt = cleanRegion(pretracksP2, pretracksP3, pretracksP4, true);
  std::vector<l1t::KMTFTrack> cleanedDisp = cleanRegion(pretracksD2, pretracksD3, pretracksD4, false);
  if (verbose_) {
    printf(
        "Prompt pretracks 2=%d 3=%d 4=%d\n", int(pretracksP2.size()), int(pretracksP3.size()), int(pretracksP4.size()));
    printf("Cleaned Tracks Prompt=%d Displaced=%d\n", (int)cleanedPrompt.size(), (int)cleanedDisp.size());
  }

  if (verbose_ && !cleanedPrompt.empty())
    for (const auto& t : cleanedPrompt)
      if (t.id() != 0)
        printf("final cleaned sector track from all chains  track pt=%d pattern=%d rank=%d\n",
               t.ptPrompt(),
               t.hitPattern(),
               t.rankPrompt());

  sort(cleanedPrompt, true);
  sort(cleanedDisp, false);

  if (verbose_ && !cleanedPrompt.empty())
    for (const auto& t : cleanedPrompt)
      if (t.id() != 0)
        printf("final sorted sector track from all chains  track pt=%d pattern=%d rank=%d\n",
               t.ptPrompt(),
               t.hitPattern(),
               t.rankPrompt());

  return std::make_pair(cleanedPrompt, cleanedDisp);
}

void KMTF::overlapCleanTrack(l1t::KMTFTrack& source, const l1t::KMTFTrack& other, bool eq, bool vertex) {
  int rank1 = vertex ? source.rankPrompt() : source.rankDisp();
  int rank2 = vertex ? other.rankPrompt() : other.rankDisp();
  int id1 = vertex ? source.id() & 0x1 : source.id() & 0x2;
  int id2 = vertex ? other.id() & 0x1 : other.id() & 0x2;
  bool keep = true;
  unsigned int pattern = 0;
  if (id1 == 0)
    keep = false;
  else if (id1 != 0 && id2 != 0) {
    if (eq && rank1 <= rank2)
      keep = false;
    if ((!eq) && rank1 < rank2)
      keep = false;
  }

  l1t::MuonStubRefVector stubs;
  for (const auto& s1 : source.stubs()) {
    bool ok = true;
    for (const auto& s2 : other.stubs()) {
      if ((*s1) == (*s2) && (!keep))
        ok = false;
    }
    if (ok) {
      stubs.push_back(s1);
      pattern = pattern | (1 << (s1->depthRegion() - 1));
    }
  }
  source.setStubs(stubs);
  source.setHitPattern(pattern);
}

std::vector<l1t::KMTFTrack> KMTF::cleanRegion(const std::vector<l1t::KMTFTrack>& tracks2,
                                              const std::vector<l1t::KMTFTrack>& tracks3,
                                              const std::vector<l1t::KMTFTrack>& tracks4,
                                              bool vertex) {
  std::vector<l1t::KMTFTrack> cleaned2;
  for (unsigned int i = 0; i < tracks2.size(); ++i) {
    l1t::KMTFTrack source = tracks2[i];

    for (unsigned int j = 0; j < tracks2.size(); ++j) {
      if (i == j)
        continue;
      overlapCleanTrack(source, tracks2[j], false, vertex);
    }
    for (unsigned int j = 0; j < tracks3.size(); ++j) {
      overlapCleanTrack(source, tracks3[j], true, vertex);
    }
    for (unsigned int j = 0; j < tracks4.size(); ++j) {
      overlapCleanTrack(source, tracks4[j], true, vertex);
    }

    if (source.stubs().size() > 1)
      cleaned2.push_back(source);
  }

  std::vector<l1t::KMTFTrack> cleaned3;
  for (unsigned int i = 0; i < tracks3.size(); ++i) {
    l1t::KMTFTrack source = tracks3[i];

    for (unsigned int j = 0; j < tracks3.size(); ++j) {
      if (i == j)
        continue;
      overlapCleanTrack(source, tracks3[j], false, vertex);
    }
    for (unsigned int j = 0; j < tracks2.size(); ++j) {
      overlapCleanTrack(source, tracks2[j], false, vertex);
    }
    for (unsigned int j = 0; j < tracks4.size(); ++j) {
      overlapCleanTrack(source, tracks4[j], true, vertex);
    }

    if (source.stubs().size() > 1)
      cleaned3.push_back(source);
  }

  std::vector<l1t::KMTFTrack> cleaned4;
  for (unsigned int i = 0; i < tracks4.size(); ++i) {
    l1t::KMTFTrack source = tracks4[i];

    for (unsigned int j = 0; j < tracks4.size(); ++j) {
      if (i == j)
        continue;
      overlapCleanTrack(source, tracks4[j], false, vertex);
    }
    for (unsigned int j = 0; j < tracks3.size(); ++j) {
      overlapCleanTrack(source, tracks3[j], false, vertex);
    }
    for (unsigned int j = 0; j < tracks2.size(); ++j) {
      overlapCleanTrack(source, tracks2[j], false, vertex);
    }

    if (source.stubs().size() > 1)
      cleaned4.push_back(source);
  }
  uint max234 = std::max(cleaned2.size(), std::max(cleaned3.size(), cleaned4.size()));

  std::vector<l1t::KMTFTrack> output;

  for (uint i = 0; i < max234; ++i) {
    if (i < cleaned2.size())
      output.push_back(cleaned2[i]);
    if (i < cleaned3.size())
      output.push_back(cleaned3[i]);
    if (i < cleaned4.size())
      output.push_back(cleaned4[i]);
  }
  return output;
}

void KMTF::swap(std::vector<l1t::KMTFTrack>& list, int i, int j, bool vertex) {
  const l1t::KMTFTrack& track1 = list[i];
  const l1t::KMTFTrack& track2 = list[j];
  int id1 = track1.id();
  int id2 = track2.id();
  int pt1 = vertex ? track1.ptPrompt() : track1.ptDisplaced();
  int pt2 = vertex ? track2.ptPrompt() : track2.ptDisplaced();
  bool swap = false;
  if (vertex) {
    id1 = id1 & 0x1;
    id2 = id2 & 0x1;
  } else {
    id1 = id1 & 0x2;
    id2 = id2 & 0x2;
  }
  if (id1 && (!id2))
    swap = false;
  else if ((!id1) && id2)
    swap = true;
  else if (id1 && id2) {
    if (pt1 > pt2)
      swap = false;
    else
      swap = true;
  } else {
    swap = false;
  }
  if (swap) {
    l1t::KMTFTrack tmp = list[i];
    list[i] = list[j];
    list[j] = tmp;
  }
}

void KMTF::sort(std::vector<l1t::KMTFTrack>& in, bool vertex) {
  l1t::KMTFTrack nullTrack;
  nullTrack.setPtEtaPhi(0, 0, 0);
  nullTrack.setIDFlag(false, false);
  nullTrack.setRank(0, vertex);
  while (in.size() < 32)
    in.push_back(nullTrack);

  for (uint iteration = 0; iteration < 16; ++iteration) {
    for (uint i = 0; i < 32; i = i + 2) {
      swap(in, i, i + 1, vertex);
    }
    for (uint i = 1; i < 31; i = i + 2) {
      swap(in, i, i + 1, vertex);
    }
  }

  std::vector<l1t::KMTFTrack> out;
  for (const auto& track : in) {
    if ((vertex && (track.id() & 0x1)) || ((!vertex) && (track.id() & 0x2)))
      out.push_back(track);
  }
  in = out;
}

class SeedSorter {
public:
  SeedSorter() {}
  bool operator()(const l1t::MuonStubRef& a, const l1t::MuonStubRef& b) { return (a->id() < b->id()); }
};
