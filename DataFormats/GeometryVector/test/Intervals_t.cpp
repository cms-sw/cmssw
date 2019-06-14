#include "DataFormats/GeometryVector/interface/PhiInterval.h"
#include "DataFormats/GeometryVector/interface/EtaInterval.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/Math/interface/PtEtaPhiMass.h"

#include <cassert>

int main() {
  assert(!checkPhiInRange(-0.9f, 1.f, -1.f));
  assert(!checkPhiInRange(0.9f, 1.f, -1.f));
  assert(checkPhiInRange(-1.1f, 1.f, -1.f));
  assert(checkPhiInRange(1.1f, 1.f, -1.f));

  assert(checkPhiInRange(-0.9f, -1.f, 1.f));
  assert(checkPhiInRange(0.9f, -1.f, 1.f));
  assert(!checkPhiInRange(-1.1f, -1.f, 1.f));
  assert(!checkPhiInRange(1.1f, -1.f, 1.f));

  assert(checkPhiInRange(-2.9f, -3.f, 3.f));
  assert(checkPhiInRange(2.9f, -3.f, 3.f));
  assert(!checkPhiInRange(-3.1f, -3.f, 3.f));
  assert(!checkPhiInRange(3.1f, -3.f, 3.f));

  assert(!checkPhiInRange(-2.9f, 3.f, -3.f));
  assert(!checkPhiInRange(2.9f, 3.f, -3.f));
  assert(checkPhiInRange(-3.1f, 3.f, -3.f));
  assert(checkPhiInRange(3.1f, 3.f, -3.f));

  for (float x = -10; x < 10; x += 1.)
    for (float y = -10; y < 10; y += 1.)
      for (float z = -10; z < 10; z += 1.) {
        if (x == 0 && y == 0)
          continue;
        GlobalPoint p(x, y, z);

        // eta
        for (float eta = -4; eta < 3.5; eta += 0.2)
          for (float deta = 0.1; deta < 2.; deta += 0.2) {
            EtaInterval ei(eta, eta + deta);
            auto in = ei.inside(p.basicVector());
            auto e = etaFromXYZ(x, y, z);
            auto in2 = (e > eta) & (e < eta + deta);
            assert(in == in2);
          }

        //phi
        for (float phi = -6.001; phi < 6.5; phi += 0.2)
          for (float dphi = -3.1; dphi < 3.15; dphi += 0.2) {
            PhiInterval pi(phi, phi + dphi);
            auto in = pi.inside(p.basicVector());
            auto ph = p.barePhi();
            auto in2 = checkPhiInRange(ph, phi, phi + dphi);
            assert(in == in2);
          }
        {
          PhiInterval pi(3.f, -3.f);
          auto in = pi.inside(p.basicVector());
          auto ph = p.barePhi();
          auto it = ph > 3.f || ph < -3.f;
          assert(in == it);
          auto in2 = checkPhiInRange(ph, 3.f, -3.f);
          assert(in2 == it);
        }
        {
          PhiInterval pi(3.f, 4.f);
          auto in = pi.inside(p.basicVector());
          auto ph = p.barePhi();
          auto it = ph > 3.f || ph < 4 - 2 * M_PI;
          ;
          assert(in == it);
          auto in2 = checkPhiInRange(ph, 3.f, 4.f);
          assert(in2 == it);
        }

        {
          PhiInterval pi(-1.f, 1.f);
          auto in = pi.inside(p.basicVector());
          auto ph = p.barePhi();
          auto it = std::abs(ph) < 1;
          assert(in == it);
          auto in2 = checkPhiInRange(ph, -1.f, 1.f);
          assert(in2 == it);
        }
      }
  return 0;
}
