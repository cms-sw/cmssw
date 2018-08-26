#include <limits>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerClusterIdentificationBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"



class HGCalTriggerClusterIdentificationBDT : public HGCalTriggerClusterIdentificationBase
{

  public:
    class Category
    {
      public:
        Category(float pt_min, float pt_max, float eta_min, float eta_max):
          pt_min_(pt_min), pt_max_(pt_max), 
          eta_min_(eta_min), eta_max_(eta_max)
        {
        }
        ~Category(){}
        bool contains(float pt, float eta) const
        {
          bool output = true;
          if(pt<pt_min_ || pt>=pt_max_) output = false;
          if(std::abs(eta)<eta_min_ || std::abs(eta)>=eta_max_) output = false;
          return output;
        }

        float pt_min() const {return pt_min_;}
        float pt_max() const {return pt_max_;}
        float eta_min() const {return eta_min_;}
        float eta_max() const {return eta_max_;}

      private:
        float pt_min_ = 0.;
        float pt_max_ = std::numeric_limits<float>::max();
        float eta_min_ = 1.5;
        float eta_max_ = 3.;
    };

  public:

    HGCalTriggerClusterIdentificationBDT();
    ~HGCalTriggerClusterIdentificationBDT() override{};
    void initialize(const edm::ParameterSet& conf) final;
    float value(const l1t::HGCalMulticluster& cluster) const final;
    bool decision(const l1t::HGCalMulticluster& cluster) const final;

  private:
    enum class ClusterVariable
    {
      cl3d_showerlength,
      cl3d_coreshowerlength,
      cl3d_firstlayer,
      cl3d_maxlayer,
      cl3d_seetot,
      cl3d_seemax,
      cl3d_spptot,
      cl3d_sppmax,
      cl3d_szz,
      cl3d_srrtot,
      cl3d_srrmax,
      cl3d_srrmean
    };
    std::vector<Category> categories_;
    std::vector<std::unique_ptr<TMVAEvaluator>> bdts_;
    std::vector<double> working_points_;
    std::vector<std::string> input_variables_;
    std::vector<ClusterVariable> input_variables_id_;

    float clusterVariable(ClusterVariable, const l1t::HGCalMulticluster&) const;
    int category(float pt, float eta) const;


};

DEFINE_HGC_TPG_CLUSTER_ID(HGCalTriggerClusterIdentificationBDT,"HGCalTriggerClusterIdentificationBDT" );


HGCalTriggerClusterIdentificationBDT::
HGCalTriggerClusterIdentificationBDT():HGCalTriggerClusterIdentificationBase()
{
}

void
HGCalTriggerClusterIdentificationBDT::
initialize(const edm::ParameterSet& conf)
{
  if(!bdts_.empty())
  {
    edm::LogWarning("HGCalTriggerClusterIdentificationBDT|Initialization")
      << "BDTs already initialized.";
    return; 
  }
  input_variables_ = conf.getParameter< std::vector<std::string> >("Inputs");
  std::vector<std::string> bdt_files = conf.getParameter< std::vector<std::string> >("Weights");
  std::vector<double> categories_etamin = conf.getParameter<std::vector<double>>("CategoriesEtaMin");
  std::vector<double> categories_etamax = conf.getParameter<std::vector<double>>("CategoriesEtaMax");
  std::vector<double> categories_ptmin = conf.getParameter<std::vector<double>>("CategoriesPtMin");
  std::vector<double> categories_ptmax = conf.getParameter<std::vector<double>>("CategoriesPtMax");
  working_points_ = conf.getParameter<std::vector<double>>("WorkingPoints");

  if(bdt_files.size()!=categories_etamin.size() ||
      categories_etamin.size()!=categories_etamax.size() ||
      categories_etamax.size()!=categories_ptmin.size() ||
      categories_ptmin.size()!=categories_ptmax.size() ||
      categories_ptmax.size()!=working_points_.size()
    )
  {
    throw cms::Exception("HGCalTriggerClusterIdentificationBDT|BadInitialization")
      << "Inconsistent numbers of categories, BDT weight files and working points";
  }
  categories_.reserve(working_points_.size());
  bdts_.reserve(working_points_.size());
  for(unsigned cat=0; cat<categories_etamin.size(); cat++)
  {
    categories_.emplace_back(
        categories_ptmin[cat],
        categories_ptmax[cat],
        categories_etamin[cat],
        categories_etamax[cat]);
  }
  std::vector<std::string> spectators = {};
  for (const auto& file : bdt_files)
  {
    bdts_.emplace_back(new TMVAEvaluator());
    bdts_.back()->initialize(
        "!Color:Silent:!Error",
        "BDT::bdt",
        edm::FileInPath(file).fullPath(),
        input_variables_,
        spectators,
        false, false);
  }

  // Transform input variable strings to enum values for later comparisons
  input_variables_id_.reserve(input_variables_.size());
  for(const auto& variable : input_variables_)
  {
    if(variable=="cl3d_showerlength") input_variables_id_.push_back(ClusterVariable::cl3d_showerlength);
    else if(variable=="cl3d_coreshowerlength") input_variables_id_.push_back(ClusterVariable::cl3d_coreshowerlength);
    else if(variable=="cl3d_firstlayer") input_variables_id_.push_back(ClusterVariable::cl3d_firstlayer);
    else if(variable=="cl3d_maxlayer") input_variables_id_.push_back(ClusterVariable::cl3d_maxlayer);
    else if(variable=="cl3d_seetot") input_variables_id_.push_back(ClusterVariable::cl3d_seetot);
    else if(variable=="cl3d_seemax") input_variables_id_.push_back(ClusterVariable::cl3d_seemax);
    else if(variable=="cl3d_spptot") input_variables_id_.push_back(ClusterVariable::cl3d_spptot);
    else if(variable=="cl3d_sppmax") input_variables_id_.push_back(ClusterVariable::cl3d_sppmax);
    else if(variable=="cl3d_szz") input_variables_id_.push_back(ClusterVariable::cl3d_szz);
    else if(variable=="cl3d_srrtot") input_variables_id_.push_back(ClusterVariable::cl3d_srrtot);
    else if(variable=="cl3d_srrmax") input_variables_id_.push_back(ClusterVariable::cl3d_srrmax);
    else if(variable=="cl3d_srrmean") input_variables_id_.push_back(ClusterVariable::cl3d_srrmean);
  }
}

float
HGCalTriggerClusterIdentificationBDT::
value(const l1t::HGCalMulticluster& cluster) const
{
  std::map<std::string, float> inputs;
  for(unsigned i=0; i<input_variables_.size(); i++)
  {
    inputs[input_variables_[i]] = clusterVariable(input_variables_id_[i], cluster);
  }
  float pt = cluster.pt();
  float eta = cluster.eta();
  int cat = category(pt, eta);
  return (cat!=-1 ? bdts_.at(cat)->evaluate(inputs) : -999.);
}


bool
HGCalTriggerClusterIdentificationBDT::
decision(const l1t::HGCalMulticluster& cluster) const
{
  float bdt_output = value(cluster);
  float pt = cluster.pt();
  float eta = cluster.eta();
  int cat = category(pt, eta);
  return (cat!=-1 ? bdt_output>working_points_.at(cat) : true);
}

int 
HGCalTriggerClusterIdentificationBDT::
category(float pt, float eta) const
{
  for(unsigned cat=0; cat<categories_.size(); cat++)
  {
    if(categories_[cat].contains(pt, eta)) return static_cast<int>(cat);
  }
  return -1;
}




float
HGCalTriggerClusterIdentificationBDT::
clusterVariable(ClusterVariable variable, const l1t::HGCalMulticluster& cluster) const
{
  switch(variable)
  {
    case ClusterVariable::cl3d_showerlength: return cluster.showerLength();
    case ClusterVariable::cl3d_coreshowerlength: return cluster.coreShowerLength();
    case ClusterVariable::cl3d_firstlayer: return cluster.firstLayer();
    case ClusterVariable::cl3d_maxlayer: return cluster.maxLayer();
    case ClusterVariable::cl3d_seetot: return cluster.sigmaEtaEtaTot();
    case ClusterVariable::cl3d_seemax: return cluster.sigmaEtaEtaMax();
    case ClusterVariable::cl3d_spptot: return cluster.sigmaPhiPhiTot();
    case ClusterVariable::cl3d_sppmax: return cluster.sigmaPhiPhiMax();
    case ClusterVariable::cl3d_szz: return cluster.sigmaZZ();
    case ClusterVariable::cl3d_srrtot: return cluster.sigmaRRTot();
    case ClusterVariable::cl3d_srrmax: return cluster.sigmaRRMax();
    case ClusterVariable::cl3d_srrmean: return cluster.sigmaRRMean();
    default: break;
  }
  return 0.;
}
