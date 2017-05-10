#include "Calibration/Tools/interface/DRings.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

void DRings::setEERings(const char * filename)
{
        FILE * fd = fopen(filename,"r");
        if (fd == NULL) {
                fprintf(stderr, "[DRings] cannot open file `%s'\n", filename);
                exit(1);
        }
        int ix = 0, iy = 0, iz = 0, ir = 0;
        while (fscanf(fd, "(%d,%d,%d) %d \n", &ix, &iy, &iz, &ir) != EOF) {
                if (iz < 0) iz = 0;
                eeRings_[ix][iy][iz] = ir;
        }
        init_ = true;
}

int DRings::ring(DetId id)
{
        assert(init_);
        if (id.subdetId() == EcalBarrel) {
                return EBDetId(id).ieta();
        } else if (id.subdetId() == EcalEndcap) {
                EEDetId eeid(id);
                int iz = eeid.zside();
                return eeRings_[eeid.ix()][eeid.iy()][iz < 0 ? 0 : iz];
        } else {
                fprintf(stderr, "[DRings::rings] invalid DetId `%d'\n", id.rawId());
        }
        return -1;
}

int DRings::ieta(DetId id)
{
        if (id.subdetId() == EcalBarrel) {
                return EBDetId(id).ieta();
        } else if (id.subdetId() == EcalEndcap) {
                EEDetId eeid(id);
                return eeid.zside() * (86 + ring(id));
        } else {
                fprintf(stderr, "[DRings::ieta] invalid DetId `%d'\n", id.rawId());
        }
        return -1;
}
