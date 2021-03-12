#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"
#include <cmath>

using Rfit::Matrix5d;
using Rfit::Vector5d;

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public MagneticField {
    M5T() : mf(0., 0., 5.) {}
    virtual GlobalVector inTesla(const GlobalPoint&) const { return mf; }

    GlobalVector mf;
  };

}  // namespace

// old pixeltrack version...
Matrix5d transfFast(Matrix5d cov, Vector5d const& p) {
  auto sqr = [](auto x) { return x * x; };
  auto sinTheta = 1 / std::sqrt(1 + p(3) * p(3));
  auto cosTheta = p(3) * sinTheta;
  cov(2, 2) = sqr(sinTheta) * (cov(2, 2) * sqr(1. / (p(2) * p(2))) + cov(3, 3) * sqr(cosTheta * sinTheta / p(2)));
  cov(3, 2) = cov(2, 3) = cov(3, 3) * cosTheta * sqr(sinTheta) / p(2);
  // for (int i=0; i<5; ++i) cov(i,2) *= -sinTheta/(p(2)*p(2));
  // for (int i=0; i<5; ++i) cov(2,i) *= -sinTheta/(p(2)*p(2));
  return cov;
}

Matrix5d loadCov(Vector5d const& e) {
  Matrix5d cov;
  for (int i = 0; i < 5; ++i)
    cov(i, i) = e(i) * e(i);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3 * std::sqrt(cov(i, i) * cov(j, j));  // this makes the matrix pos defined
      cov(i, j) = (i + j) % 2 ? -0.4 * v : 0.1 * v;
      cov(j, i) = cov(i, j);
    }
  }
  return cov;
}

#include <iostream>
int main() {
  M5T const mf;

  for (auto charge = -1; charge < 2; charge += 2)
    for (auto szip = -1; szip < 2; szip += 2)
      for (auto stip = -1; stip < 2; stip += 2) {
        Vector5d par0;
        par0 << 0.2, 0.1, 3.5, 0.8, 0.1;
        Vector5d del0;
        del0 << 0.01, 0.01, 0.035, -0.03, -0.01;
        //!<(phi,Tip,pt,cotan(theta)),Zip)
        par0(1) *= stip;
        par0(4) *= szip;

        Matrix5d cov0 = loadCov(del0);

        Vector5d par1;
        Vector5d par2;

        Matrix5d cov1;
        Matrix5d cov2;

        // Matrix5d covf = transfFast(cov0,par0);

        Rfit::transformToPerigeePlane(par0, cov0, par1, cov1);

        std::cout << "cov1\n" << cov1 << std::endl;

        LocalTrajectoryParameters lpar(par1(0), par1(1), par1(2), par1(3), par1(4), 1.);
        AlgebraicSymMatrix55 m;
        for (int i = 0; i < 5; ++i)
          for (int j = i; j < 5; ++j)
            m(i, j) = cov1(i, j);

        float phi = par0(0);
        float sp = std::sin(phi);
        float cp = std::cos(phi);
        Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

        Surface::PositionType bs(0., 0., 0.);
        Plane plane(bs, rot);
        GlobalTrajectoryParameters gp(
            plane.toGlobal(lpar.position()), plane.toGlobal(lpar.momentum()), lpar.charge(), &mf);
        std::cout << "global par " << gp.position() << ' ' << gp.momentum() << ' ' << gp.charge() << std::endl;
        JacobianLocalToCurvilinear jl2c(plane, lpar, mf);
        std::cout << "jac l2c" << jl2c.jacobian() << std::endl;

        AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);
        std::cout << "curv error\n" << mo << std::endl;

        /*

  // not accurate as the perigee plane move as well...
  Vector5d del1 = par2-par1;


  // don't ask: guess
  std::cout << "charge " << charge << std::endl;
  std::cout << "par0 " << par0.transpose() << std::endl;
  std::cout << "del0 " << del0.transpose() << std::endl;


  std::cout << "par1 " << par1.transpose() << std::endl;
  std::cout << "del1 " << del1.transpose() << std::endl;
  // std::cout << "del2 " << (J*del0).transpose() << std::endl;

  std::cout << "del1^2 " << (del1.array()*del1.array()).transpose() << std::endl;
  std::cout << std::endl;
  
  std::cout << "cov0\n" << cov0 << std::endl;
  std::cout << "cov1\n" << cov1 << std::endl;
  std::cout << "cov2\n" << cov2 << std::endl;
  */

        std::cout << std::endl << "----------" << std::endl;

      }  // lopp over signs

  return 0;
}
