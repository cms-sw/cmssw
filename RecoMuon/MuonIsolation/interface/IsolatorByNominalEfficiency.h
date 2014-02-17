#ifndef MuonIsolation_IsolatorByNominalEfficiency_H
#define MuonIsolation_IsolatorByNominalEfficiency_H

/** \class IsolatorByNominalEfficiency
 *  Computes the isolation variable as "nominal efficiency",
 *  defined so that a cut at a value X will give X efficiency
 *  on the reference signal (W->munu). See CMS/Note 2002/040 for details.
 *
 *  $Date: 2012/01/27 06:02:20 $
 *  \author M. Konecki, N. Amapane
 */

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"
#include <vector>
#include <string>

namespace muonisolation { class NominalEfficiencyThresholds; }

namespace muonisolation {
  class IsolatorByNominalEfficiency : public MuIsoBaseIsolator {
  public:
    typedef MuIsoBaseIsolator::DepositContainer DepositContainer;

    //! Constructor
    IsolatorByNominalEfficiency(const std::string & thrFile,
				const std::vector<std::string> & ceff,
				const std::vector<double> & weights);

    IsolatorByNominalEfficiency(const std::string & thrFile,
				const std::vector<std::string> & ceff,
				const std::vector<double> & weights, const std::vector<double> & thresh	);


    virtual ~IsolatorByNominalEfficiency();

    //! Compute the deposit within the cone and return the isolation result
    virtual Result result(const DepositContainer& deposits, const edm::Event* = 0) const;

    Cuts cuts(float nominalEfficiency) const;

  private:

    class ConeSizes {
    private:
      enum  IsoDim { DIM = 15};
      static float cone_dr[DIM];
    public:
      int dim() const { return DIM;}
      double size(int i) const;
      int index(float dr) const;
    };

    // Compute the weighted sum of deposits of different type within dRcone
    virtual double weightedSum(const DepositContainer& deposits, float dRcone) const;

    // Size of cone for a given nominal efficiency value.
    int bestConeForEfficiencyIndex(float eff_thr) const;

    typedef std::multimap<float,int> mapNomEff_Cone;
    mapNomEff_Cone cones(const std::vector<std::string>& names);

    std::string findPath(const std::string& fileName);

    virtual ResultType resultType() const { return ISOL_FLOAT_TYPE;}

  private:
    mapNomEff_Cone coneForEfficiency;
    NominalEfficiencyThresholds * thresholds;
    std::vector<double> theWeights;
    std::vector<double> theDepThresholds;
    ConeSizes theConesInfo;
  };
}

#endif
