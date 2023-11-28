#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
  /// is sv1 < sv2
  struct ddsvaluesCmp {
    bool operator()(const DDsvalues_type& sv1, const DDsvalues_type& sv2) const;
  };
}  // namespace

class OutputDDToDDL : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit OutputDDToDDL(const edm::ParameterSet& iConfig);
  ~OutputDDToDDL() override;

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void endJob() override {}

private:
  void addToMatStore(const DDMaterial& mat, std::set<DDMaterial>& matStore);
  void addToSolStore(const DDSolid& sol, std::set<DDSolid>& solStore, std::set<DDRotation>& rotStore);
  void addToSpecStore(const DDLogicalPart& lp,
                      std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp>& specStore);

  int m_rotNumSeed;
  std::string m_fname;
  std::ostream* m_xos;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
};

bool ddsvaluesCmp::operator()(const DDsvalues_type& sv1, const DDsvalues_type& sv2) const {
  if (sv1.size() < sv2.size())
    return true;
  if (sv2.size() < sv1.size())
    return false;

  size_t ind = 0;
  for (; ind < sv1.size(); ++ind) {
    if (sv1[ind].first < sv2[ind].first)
      return true;
    if (sv2[ind].first < sv1[ind].first)
      return false;
    if (sv1[ind].second < sv2[ind].second)
      return true;
    if (sv2[ind].second < sv1[ind].second)
      return false;
  }
  return false;
}

OutputDDToDDL::OutputDDToDDL(const edm::ParameterSet& iConfig) : m_fname() {
  m_rotNumSeed = iConfig.getParameter<int>("rotNumSeed");
  m_fname = iConfig.getUntrackedParameter<std::string>("fileName");
  if (m_fname.empty()) {
    m_xos = &std::cout;
  } else {
    m_xos = new std::ofstream(m_fname.c_str());
  }
  (*m_xos) << "<?xml version=\"1.0\"?>" << std::endl;
  (*m_xos) << "<DDDefinition xmlns=\"http://www.cern.ch/cms/DDL\"" << std::endl;
  (*m_xos) << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" << std::endl;
  (*m_xos) << "xsi:schemaLocation=\"http://www.cern.ch/cms/DDL ../../../DetectorDescription/Schema/DDLSchema.xsd\">"
           << std::endl;
  (*m_xos) << std::fixed << std::setprecision(18);

  ddToken_ = esConsumes<DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
}

OutputDDToDDL::~OutputDDToDDL() {
  (*m_xos) << "</DDDefinition>" << std::endl;
  (*m_xos) << std::endl;
  m_xos->flush();
}

void OutputDDToDDL::beginRun(const edm::Run&, edm::EventSetup const& es) {
  std::cout << "OutputDDToDDL::beginRun" << std::endl;

  edm::ESTransientHandle<DDCompactView> pDD = es.getTransientHandle(ddToken_);

  using Graph = DDCompactView::Graph;
  using adjl_iterator = Graph::const_adj_iterator;

  const auto& gra = pDD->graph();
  // temporary stores:
  std::set<DDLogicalPart> lpStore;
  std::set<DDMaterial> matStore;
  std::set<DDSolid> solStore;
  // 2009-08-19: MEC: I've tried this with set<DDPartSelection> and
  // had to write operator< for DDPartSelection and DDPartSelectionLevel
  // the output from such an effort is different than this one.
  std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp> specStore;
  std::set<DDRotation> rotStore;

  DDCoreToDDXMLOutput out;

  std::string rn = m_fname;
  size_t foundLastDot = rn.find_last_of('.');
  size_t foundLastSlash = rn.find_last_of('/');
  if (foundLastSlash > foundLastDot && foundLastSlash != std::string::npos) {
    std::cout << "What? last . before last / in path for filename... this should die..." << std::endl;
  }
  if (foundLastDot != std::string::npos && foundLastSlash != std::string::npos) {
    out.ns_ = rn.substr(foundLastSlash, foundLastDot);
  } else if (foundLastDot != std::string::npos) {
    out.ns_ = rn.substr(0, foundLastDot);
  } else {
    std::cout << "What? no file name? Attempt at namespace =\"" << out.ns_ << "\" filename was " << m_fname
              << std::endl;
  }
  std::cout << "m_fname = " << m_fname << " namespace = " << out.ns_ << std::endl;
  std::string ns_ = out.ns_;

  (*m_xos) << std::fixed << std::setprecision(18);

  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();

  (*m_xos) << "<PosPartSection label=\"" << ns_ << "\">" << std::endl;
  git = gra.begin();
  for (; git != gend; ++git) {
    const DDLogicalPart& ddLP = gra.nodeData(git);
    if (lpStore.find(ddLP) != lpStore.end()) {
      addToSpecStore(ddLP, specStore);
    }
    lpStore.insert(ddLP);
    addToMatStore(ddLP.material(), matStore);
    addToSolStore(ddLP.solid(), solStore, rotStore);
    if (!git->empty()) {
      // ask for children of ddLP
      auto cit = git->begin();
      auto cend = git->end();
      for (; cit != cend; ++cit) {
        const DDLogicalPart& ddcurLP = gra.nodeData(cit->first);
        if (lpStore.find(ddcurLP) != lpStore.end()) {
          addToSpecStore(ddcurLP, specStore);
        }
        lpStore.insert(ddcurLP);
        addToMatStore(ddcurLP.material(), matStore);
        addToSolStore(ddcurLP.solid(), solStore, rotStore);
        rotStore.insert(gra.edgeData(cit->second)->ddrot());
        out.position(ddLP, ddcurLP, gra.edgeData(cit->second), m_rotNumSeed, *m_xos);
      }  // iterate over children
    }    // if (children)
  }      // iterate over graph nodes

  (*m_xos) << "</PosPartSection>" << std::endl;

  (*m_xos) << std::scientific << std::setprecision(18);

  (*m_xos) << "<MaterialSection label=\"" << ns_ << "\">" << std::endl;
  for (const auto& it : matStore) {
    if (!it.isDefined().second)
      continue;
    out.material(it, *m_xos);
  }
  (*m_xos) << "</MaterialSection>" << std::endl;
  (*m_xos) << "<RotationSection label=\"" << ns_ << "\">" << std::endl;
  (*m_xos) << std::fixed << std::setprecision(18);
  std::set<DDRotation>::iterator rit(rotStore.begin()), red(rotStore.end());
  for (; rit != red; ++rit) {
    if (!rit->isDefined().second)
      continue;
    if (rit->toString() != ":") {
      const DDRotation& r(*rit);
      out.rotation(r, *m_xos);
    }
  }
  (*m_xos) << "</RotationSection>" << std::endl;

  (*m_xos) << std::fixed << std::setprecision(18);
  std::set<DDSolid>::const_iterator sit(solStore.begin()), sed(solStore.end());
  (*m_xos) << "<SolidSection label=\"" << ns_ << "\">" << std::endl;
  for (; sit != sed; ++sit) {
    if (!sit->isDefined().second)
      continue;
    out.solid(*sit, *m_xos);
  }
  (*m_xos) << "</SolidSection>" << std::endl;

  std::set<DDLogicalPart>::iterator lpit(lpStore.begin()), lped(lpStore.end());
  (*m_xos) << "<LogicalPartSection label=\"" << ns_ << "\">" << std::endl;
  for (; lpit != lped; ++lpit) {
    if (!lpit->isDefined().first)
      continue;
    const DDLogicalPart& lp = *lpit;
    out.logicalPart(lp, *m_xos);
  }
  (*m_xos) << "</LogicalPartSection>" << std::endl;

  (*m_xos) << std::fixed << std::setprecision(18);
  std::map<DDsvalues_type, std::set<const DDPartSelection*> >::const_iterator mit(specStore.begin()),
      mend(specStore.end());
  (*m_xos) << "<SpecParSection label=\"" << ns_ << "\">" << std::endl;
  for (; mit != mend; ++mit) {
    out.specpar(*mit, *m_xos);
  }
  (*m_xos) << "</SpecParSection>" << std::endl;
}

void OutputDDToDDL::addToMatStore(const DDMaterial& mat, std::set<DDMaterial>& matStore) {
  matStore.insert(mat);
  if (mat.noOfConstituents() != 0) {
    int findex(0);
    while (findex < mat.noOfConstituents()) {
      if (matStore.find(mat.constituent(findex).first) == matStore.end()) {
        addToMatStore(mat.constituent(findex).first, matStore);
      }
      ++findex;
    }
  }
}

void OutputDDToDDL::addToSolStore(const DDSolid& sol, std::set<DDSolid>& solStore, std::set<DDRotation>& rotStore) {
  solStore.insert(sol);
  if (sol.shape() == DDSolidShape::ddunion || sol.shape() == DDSolidShape::ddsubtraction ||
      sol.shape() == DDSolidShape::ddintersection) {
    const DDBooleanSolid& bs(sol);
    if (solStore.find(bs.solidA()) == solStore.end()) {
      addToSolStore(bs.solidA(), solStore, rotStore);
    }
    if (solStore.find(bs.solidB()) == solStore.end()) {
      addToSolStore(bs.solidB(), solStore, rotStore);
    }
    rotStore.insert(bs.rotation());
  }
}

void OutputDDToDDL::addToSpecStore(
    const DDLogicalPart& lp,
    std::map<const DDsvalues_type, std::set<const DDPartSelection*>, ddsvaluesCmp>& specStore) {
  for (auto spit : lp.attachedSpecifics()) {
    specStore[*spit.second].insert(spit.first);
  }
}

DEFINE_FWK_MODULE(OutputDDToDDL);
