#include "Calibration/Tools/interface/IC.h"
#include "Calibration/Tools/interface/DS.h"

std::vector<DetId> IC::_detId;
DRings IC::dr_;
bool IC::idr_;

IC::IC()
{
        _detId.resize(EBDetId::MAX_HASH + 1 + EEDetId::kSizeForDenseIndexing);
        int idx = -1;
        for (int hi = EBDetId::MIN_HASH; hi <= EBDetId::MAX_HASH; ++hi ) {
                EBDetId ebId = EBDetId::unhashIndex(hi);
                if (ebId != EBDetId()) {
                        _detId[hi] = ebId;
                }
        }
        for ( int hi = 0; hi < EEDetId::kSizeForDenseIndexing; ++hi ) {
                EEDetId eeId = EEDetId::unhashIndex(hi);
                if (eeId != EEDetId()) {
                        idx = EBDetId::MAX_HASH + 1 + hi;
                        _detId[idx] = eeId;
                }
        }
        assert(_detId.size() == 75848);
        assert(_detId.size() == EBDetId::MAX_HASH + 1 + EEDetId::kSizeForDenseIndexing);
}

void IC::coord(DetId id, Coord * c)
{
        if (id.subdetId() == EcalBarrel) {
                EBDetId eid(id);
                c->ix_ = eid.ieta();
                c->iy_ = eid.iphi();
                c->iz_ = 0;
        } else if (id.subdetId() == EcalEndcap) {
                EEDetId eid(id);
                c->ix_ = eid.ix();
                c->iy_ = eid.iy();
                c->iz_ = eid.zside();
        } else {
                fprintf(stderr, "[IC::coord] ERROR: invalid DetId %d", id.rawId());        
                assert(0);
        }
}



void IC::constantDistribution(const IC & a, TH1F * h, DS & selector, bool errors)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                if (errors) h->Fill(a.eic()[id]);
                else        h->Fill(a.ic()[id]);
        }
}


void IC::constantMap(const IC & a, TH2F * h, DS & selector, bool errors)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int ix = 0, iy = 0;
                if (id.subdetId() == EcalBarrel) {
                        ix = EBDetId(id).iphi();
                        iy = EBDetId(id).ieta();
                } else if (id.subdetId() == EcalEndcap) {
                        ix = EEDetId(id).ix();
                        iy = EEDetId(id).iy();
                }
                if (errors) h->Fill(ix, iy, a.eic()[id]);
                else        h->Fill(ix, iy, a.ic()[id]);
        }
}


void IC::profileEta(const IC & a, TProfile * h, DS & selector, bool errors)
{
        if (!idr_) {
                fprintf(stderr, "Please set the DRings object to your IC object\n");
                fprintf(stderr, "before calling `profileEta(...)', e.g. for C++\n");
                fprintf(stderr, "DRings dr;\n");
                fprintf(stderr, "dr.setEERings(\"eerings.dat\");\n");
                fprintf(stderr, "ic.setRings(rings);\n");
                assert(0);
        }
        //DRings dr;
        //dr.setEERings("eerings.dat");
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int x = dr_.ieta(id);
                //printf("ieta %d\n", dr.ieta(id));
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                if (errors) h->Fill(x, e);
                else        h->Fill(x, v);
        }
}


void IC::profilePhi(const IC & a, TProfile * h, DS & selector, bool errors)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int x = 0;
                if (id.subdetId() == EcalBarrel) {
                        x = EBDetId(id).iphi();
                } else if (id.subdetId() == EcalEndcap) {
                        //x = atan((EEDetId(id).iy() - 50) / (EEDetId(id).ix() - 50));
                }
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                if (errors) h->Fill(x, e);
                else        h->Fill(x, v);
        }
}


void IC::profileSM(const IC & a, TProfile * h, DS & selector, bool errors)
{
        DSIsBarrel isBarrel;
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id) || !isBarrel(id)) continue;
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                if (errors) h->Fill(EBDetId(id).ism(), e);
                else        h->Fill(EBDetId(id).ism(), v);
        }
}


float IC::average(const IC & a, DS & selector, bool errors)
{
        float ave = 0.;
        int cnt = 0;
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                ++cnt;
                if (errors) ave += e;
                else        ave += v;
        }
        assert(cnt != 0);
        return ave / cnt;
}


void IC::reciprocal(const IC & a, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                res.ic().setValue(id, 1. / v);
                res.eic().setValue(id, e / v / v);
        }
}


void IC::multiply(const IC & a, float c, IC & res, DS & selector)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                res.ic().setValue(id, a.ic()[id] * c);
                res.eic().setValue(id, a.eic()[id] * fabs(c));
        }
}


void IC::multiply(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                res.ic().setValue(id, va * vb);
                res.eic().setValue(id, sqrt(vb * vb * ea * ea + va * va * eb * eb));
        }
}


void IC::add(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                res.ic().setValue(id, va + vb);
                res.eic().setValue(id, sqrt(ea * ea + eb * eb));
        }
}


void IC::combine(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                if (isValid(va, ea) && isValid(vb, eb)) {
                        float wa = 1. / ea / ea;
                        float wb = 1. / eb / eb;
                        res.ic().setValue(id, (wa * va + wb * vb) / (wa + wb));
                        res.eic().setValue(id, 1. / sqrt(wa + wb));
                } else if (isValid(va, ea)) {
                        res.ic().setValue(id, va);
                        res.eic().setValue(id, ea);
                } else if (isValid(vb, eb)) {
                        res.ic().setValue(id, vb);
                        res.eic().setValue(id, eb);
                } else {
                        res.ic().setValue(id, 1);
                        res.eic().setValue(id, 999);
                }
        }
}


void IC::smear(const IC & a, float sigma, IC & res)
{
        TRandom * r = new TRandom();
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float s = r->Gaus(1, sigma);
                res.ic().setValue(id, a.ic()[id] * s);
                res.eic().setValue(id, a.eic()[id] * s);
        }
        delete r;
}


void IC::smear(const IC & a, IC & res)
{
        TRandom * r = new TRandom();
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float s = r->Gaus(1, a.eic()[id]);
                res.ic().setValue(id, a.ic()[id] * s);
                res.eic().setValue(id, a.eic()[id]);
        }
        delete r;
}


void IC::dump(const IC & a, const char * fileName, DS & selector)
{
        FILE * fd = fopen(fileName, "w");
        if (fd == NULL) {
                fprintf(stderr, "[dump] cannot open file %s\n", fileName);
                exit(-1);
        }
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                //fprintf(fd, "%d %f %f\n", id.rawId(), a.ic()[id], a.eic()[id]);
                if (id.subdetId() == EcalBarrel) {
                        EBDetId eid(id);
                        fprintf(fd, "%d %d %d %f %f\n", eid.ieta(), eid.iphi(), 0, a.ic()[id], a.eic()[id]);
                } else if (id.subdetId() == EcalEndcap) {
                        EEDetId eid(id);
                        fprintf(fd, "%d %d %d %f %f\n", eid.ix(), eid.iy(), eid.zside(), a.ic()[id], a.eic()[id]);
                } else {
                        fprintf(stderr, "[dump] invalid DetId: %d\n", id.rawId());
                        exit(-1);
                }
        }
        fclose(fd);
}



void IC::dumpEtaScale(const IC & a, const char * fileName, bool allIC)
{
        FILE * fd = fopen(fileName, "w");
        if (fd == NULL) {
                fprintf(stderr, "[dumpEtaScale] cannot open file %s\n", fileName);
                exit(-1);
        }
        fprintf(fd, "#eta scale dump\n");
        if (!allIC) fprintf(fd, "#iring  scale  rms_on_the_mean\n");
        else        fprintf(fd, "#ieta/ix iphi/iy 0/zside  scale  0\n");
        float etasum[DRings::nHalfIEta * 2 + 1]; // ieta = 0 does not exist
        float etasum2[DRings::nHalfIEta * 2 + 1]; // ieta = 0 does not exist
        int n[DRings::nHalfIEta * 2 + 1]; // ieta = 0 does not exist
        for (int i = 0; i < DRings::nHalfIEta * 2 + 1; ++i) {
                etasum[i] = 0;
                etasum2[i] = 0;
                n[i] = 0;
        }
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float v = a.ic()[id];
                float e = a.eic()[id];
                if (!isValid(v, e)) continue;
                int idx = dr_.ieta(id) + DRings::nHalfIEta;
                etasum[idx] += v;
                etasum2[idx] += v * v;
                n[idx]++;
        }
        if (!allIC) {
                for (int i = 0; i < DRings::nHalfIEta * 2 + 1; ++i) {
                        int ir = i - DRings::nHalfIEta;
                        if (ir != 0) fprintf(fd, "%d %f %f\n", ir, etasum[i] / n[i], sqrt(etasum2[i] * etasum2[i] / n[i] - etasum[i] * etasum[i] / n[i] / n[i]) / n[i]);
                }
        } else {
                for (size_t i = 0; i < a.ids().size(); ++i) {
                        DetId id(a.ids()[i]);
                        int idx = dr_.ieta(id) + DRings::nHalfIEta;
                        float v = etasum[idx] / n[idx];
                        //float e = sqrt(etasum2[idx] * etasum2[idx] / n[idx] - etasum[idx] * etasum[idx] / n[idx] / n[idx]) / n[idx];
                        float e = 0;
                        if (id.subdetId() == EcalBarrel) {
                                EBDetId eid(id);
                                fprintf(fd, "%d %d %d %f %f\n", eid.ieta(), eid.iphi(), 0, v, e);
                        } else if (id.subdetId() == EcalEndcap) {
                                EEDetId eid(id);
                                fprintf(fd, "%d %d %d %f %f\n", eid.ix(), eid.iy(), eid.zside(), v, e);
                        }
                }
        }
        fclose(fd);
}



void IC::dumpXML(const IC & a, const char * fileName, DS & selector, bool errors)
{
        FILE * fd = fopen(fileName, "w");
        if (fd == NULL) {
                fprintf(stderr, "[dumpXML] cannot open file %s\n", fileName);
                exit(-1);
        }
        // exactly the same spacing/new lines
        //
        fprintf(fd, "\n");
        fprintf(fd, "<EcalFloatCondObjectContainer>\n");
        fprintf(fd, "\n");
        fprintf(fd, "  <EcalCondHeader>\n");
        fprintf(fd, "    <method></method>\n");
        fprintf(fd, "    <version></version>\n");
        fprintf(fd, "    <datasource></datasource>\n");
        fprintf(fd, "    <since>0</since>\n");
        fprintf(fd, "    <tag></tag>\n");
        fprintf(fd, "    <date></date>\n");
        fprintf(fd, "  </EcalCondHeader>\n");
        fprintf(fd, "\n");

        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                //fprintf(fd, "%d %f %f\n", id.rawId(), a.ic()[id], a.eic()[id]);
                if (id.subdetId() == EcalBarrel) {
                        EBDetId eid(id);
                        fprintf(fd, "  <cell iEta=\"%d\" iPhi=\"%d\">\n", eid.ieta(), eid.iphi());
                        if (errors) fprintf(fd, "    <Value>%f</Value>\n", a.eic()[id]);
                        else        fprintf(fd, "    <Value>%f</Value>\n", a.ic()[id]);
                        fprintf(fd, "  </cell>\n");
                        fprintf(fd, "\n");

                } else if (id.subdetId() == EcalEndcap) {
                        EEDetId eid(id);
                        fprintf(fd, "  <cell ix=\"%d\" iy=\"%d\" zside=\"%d\">\n", eid.ix(), eid.iy(), eid.zside());
                        if (errors) fprintf(fd, "    <Value>%f</Value>\n", a.eic()[id]);
                        else        fprintf(fd, "    <Value>%f</Value>\n", a.ic()[id]);
                        fprintf(fd, "  </cell>\n");
                        fprintf(fd, "\n");
                } else {
                        fprintf(stderr, "[dump] invalid DetId: %d\n", id.rawId());
                        exit(-1);
                }
        }
        fprintf(fd, "</EcalFloatCondObjectContainer>\n");
        fclose(fd);
}



//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

class TEndcapRings {
        private:
                float iEndcapRing[100][100][2];
        public:
                TEndcapRings();
                ~TEndcapRings();
                Int_t GetEndcapRing(Int_t,Int_t,Int_t);
                Int_t GetEndcapIeta(Int_t,Int_t,Int_t);

                //ClassDef(TEndcapRings,1); //ring class
};

// default constructor, reading the map from file
TEndcapRings::TEndcapRings() {
        FILE *fRing;
        fRing = fopen("eerings.dat","r");
        int ix,iy,iz,ir;
        while(fscanf(fRing,"(%d,%d,%d) %d \n",&ix,&iy,&iz,&ir) !=EOF ) {
                if (iz<0) iz=0;
                iEndcapRing[ix][iy][iz] = ir;
        }
        return;
}

TEndcapRings::~TEndcapRings() { return;}

Int_t TEndcapRings::GetEndcapRing(Int_t ix, Int_t iy, Int_t iz){
        return iEndcapRing[ix][iy][iz];
}

Int_t TEndcapRings::GetEndcapIeta(Int_t ix, Int_t iy, Int_t iz){
        Int_t iSide = iz;
        if (iSide<0) iSide=0;
        Int_t iEtaOffset = 86*iz;
        Int_t iEta = iEtaOffset + iz*iEndcapRing[ix][iy][iSide];
        return iEta;
}

// --- Here the calibration functions
Double_t fCalEoP(Int_t ieta) {
  Double_t fCal=1.;
  if (ieta==-118) fCal = 0.963263;
  if (ieta==-117) fCal = 0.971674;
  if (ieta==-116) fCal = 0.9874;
  if (ieta==-115) fCal = 1.01517;
  if (ieta==-114) fCal = 1.00988;
  if (ieta==-113) fCal = 1.01252;
  if (ieta==-112) fCal = 0.996157;
  if (ieta==-111) fCal = 0.995526;
  if (ieta==-110) fCal = 0.992545;
  if (ieta==-109) fCal = 0.989214;
  if (ieta==-108) fCal = 0.996461;
  if (ieta==-107) fCal = 0.993015;
  if (ieta==-106) fCal = 0.983365;
  if (ieta==-105) fCal = 0.988003;
  if (ieta==-104) fCal = 0.991306;
  if (ieta==-103) fCal = 0.994093;
  if (ieta==-102) fCal = 1.00807;
  if (ieta==-101) fCal = 0.999271;
  if (ieta==-100) fCal = 0.994555;
  if (ieta==-99) fCal = 1.0025;
  if (ieta==-98) fCal = 1.00659;
  if (ieta==-97) fCal = 1.00645;
  if (ieta==-96) fCal = 1.0132;
  if (ieta==-95) fCal = 0.990159;
  if (ieta==-94) fCal = 0.979428;
  if (ieta==-93) fCal = 0.985317;
  if (ieta==-92) fCal = 0.991403;
  if (ieta==-91) fCal = 0.982159;
  if (ieta==-90) fCal = 0.968961;
  if (ieta==-89) fCal = 0.96145;
  if (ieta==-88) fCal = 0.968126;
  if (ieta==-87) fCal = 0.962001;
  if (ieta==-86) fCal = 0.951714;
  if (ieta==-85) fCal = 1.01721;
  if (ieta==-84) fCal = 1.01174;
  if (ieta==-83) fCal = 1.01161;
  if (ieta==-82) fCal = 1.00886;
  if (ieta==-81) fCal = 1.00545;
  if (ieta==-80) fCal = 1.00469;
  if (ieta==-79) fCal = 1.00612;
  if (ieta==-78) fCal = 1.00146;
  if (ieta==-77) fCal = 1.00326;
  if (ieta==-76) fCal = 1.00462;
  if (ieta==-75) fCal = 1.00733;
  if (ieta==-74) fCal = 1.00738;
  if (ieta==-73) fCal = 1.00689;
  if (ieta==-72) fCal = 1.00509;
  if (ieta==-71) fCal = 1.00758;
  if (ieta==-70) fCal = 1.01067;
  if (ieta==-69) fCal = 1.0039;
  if (ieta==-68) fCal = 1.00313;
  if (ieta==-67) fCal = 1.0072;
  if (ieta==-66) fCal = 1.00055;
  if (ieta==-65) fCal = 1.00402;
  if (ieta==-64) fCal = 1.00463;
  if (ieta==-63) fCal = 1.00334;
  if (ieta==-62) fCal = 1.00497;
  if (ieta==-61) fCal = 1.00563;
  if (ieta==-60) fCal = 1.00357;
  if (ieta==-59) fCal = 0.998981;
  if (ieta==-58) fCal = 0.999742;
  if (ieta==-57) fCal = 1.0025;
  if (ieta==-56) fCal = 1.00258;
  if (ieta==-55) fCal = 1.0022;
  if (ieta==-54) fCal = 1.00174;
  if (ieta==-53) fCal = 1.00181;
  if (ieta==-52) fCal = 0.999915;
  if (ieta==-51) fCal = 1.00011;
  if (ieta==-50) fCal = 1.00038;
  if (ieta==-49) fCal = 0.999399;
  if (ieta==-48) fCal = 1.00148;
  if (ieta==-47) fCal = 1.00016;
  if (ieta==-46) fCal = 0.999773;
  if (ieta==-45) fCal = 0.998409;
  if (ieta==-44) fCal = 0.997334;
  if (ieta==-43) fCal = 0.99616;
  if (ieta==-42) fCal = 0.996602;
  if (ieta==-41) fCal = 0.996881;
  if (ieta==-40) fCal = 0.997341;
  if (ieta==-39) fCal = 0.996775;
  if (ieta==-38) fCal = 0.997547;
  if (ieta==-37) fCal = 0.997444;
  if (ieta==-36) fCal = 0.999854;
  if (ieta==-35) fCal = 0.996914;
  if (ieta==-34) fCal = 0.996279;
  if (ieta==-33) fCal = 0.993053;
  if (ieta==-32) fCal = 0.996378;
  if (ieta==-31) fCal = 0.996194;
  if (ieta==-30) fCal = 0.99681;
  if (ieta==-29) fCal = 0.99741;
  if (ieta==-28) fCal = 0.996968;
  if (ieta==-27) fCal = 0.998254;
  if (ieta==-26) fCal = 1.00016;
  if (ieta==-25) fCal = 1.00062;
  if (ieta==-24) fCal = 0.995513;
  if (ieta==-23) fCal = 0.997131;
  if (ieta==-22) fCal = 0.998236;
  if (ieta==-21) fCal = 1.00041;
  if (ieta==-20) fCal = 0.9971;
  if (ieta==-19) fCal = 0.994644;
  if (ieta==-18) fCal = 0.99258;
  if (ieta==-17) fCal = 0.995608;
  if (ieta==-16) fCal = 0.996795;
  if (ieta==-15) fCal = 0.997996;
  if (ieta==-14) fCal = 0.994031;
  if (ieta==-13) fCal = 0.996509;
  if (ieta==-12) fCal = 0.993154;
  if (ieta==-11) fCal = 0.995278;
  if (ieta==-10) fCal = 0.995381;
  if (ieta==-9) fCal = 0.997578;
  if (ieta==-8) fCal = 0.995718;
  if (ieta==-7) fCal = 0.998568;
  if (ieta==-6) fCal = 0.997773;
  if (ieta==-5) fCal = 0.999943;
  if (ieta==-4) fCal = 0.999012;
  if (ieta==-3) fCal = 1.00206;
  if (ieta==-2) fCal = 1.00176;
  if (ieta==-1) fCal = 1.00387;
  if (ieta==1) fCal = 1.00107;
  if (ieta==2) fCal = 1.00295;
  if (ieta==3) fCal = 1.00147;
  if (ieta==4) fCal = 1.00137;
  if (ieta==5) fCal = 0.998867;
  if (ieta==6) fCal = 0.996665;
  if (ieta==7) fCal = 0.996115;
  if (ieta==8) fCal = 0.993869;
  if (ieta==9) fCal = 0.99454;
  if (ieta==10) fCal = 0.997421;
  if (ieta==11) fCal = 0.995012;
  if (ieta==12) fCal = 0.994027;
  if (ieta==13) fCal = 0.996042;
  if (ieta==14) fCal = 0.995489;
  if (ieta==15) fCal = 0.996863;
  if (ieta==16) fCal = 0.997838;
  if (ieta==17) fCal = 0.994613;
  if (ieta==18) fCal = 0.995481;
  if (ieta==19) fCal = 0.996187;
  if (ieta==20) fCal = 0.998448;
  if (ieta==21) fCal = 1.00074;
  if (ieta==22) fCal = 0.9977;
  if (ieta==23) fCal = 0.998234;
  if (ieta==24) fCal = 0.999486;
  if (ieta==25) fCal = 1.0009;
  if (ieta==26) fCal = 0.998134;
  if (ieta==27) fCal = 0.996344;
  if (ieta==28) fCal = 0.9971;
  if (ieta==29) fCal = 0.996177;
  if (ieta==30) fCal = 0.99774;
  if (ieta==31) fCal = 0.997081;
  if (ieta==32) fCal = 0.995681;
  if (ieta==33) fCal = 0.993957;
  if (ieta==34) fCal = 0.99727;
  if (ieta==35) fCal = 0.996595;
  if (ieta==36) fCal = 0.997328;
  if (ieta==37) fCal = 0.995289;
  if (ieta==38) fCal = 0.994593;
  if (ieta==39) fCal = 0.996694;
  if (ieta==40) fCal = 0.997293;
  if (ieta==41) fCal = 0.996981;
  if (ieta==42) fCal = 0.998503;
  if (ieta==43) fCal = 0.996167;
  if (ieta==44) fCal = 0.996906;
  if (ieta==45) fCal = 0.998366;
  if (ieta==46) fCal = 0.99891;
  if (ieta==47) fCal = 0.999686;
  if (ieta==48) fCal = 0.999189;
  if (ieta==49) fCal = 0.998509;
  if (ieta==50) fCal = 0.999672;
  if (ieta==51) fCal = 0.998637;
  if (ieta==52) fCal = 1.00223;
  if (ieta==53) fCal = 1.00144;
  if (ieta==54) fCal = 1.00094;
  if (ieta==55) fCal = 1.00048;
  if (ieta==56) fCal = 1.00103;
  if (ieta==57) fCal = 1.00056;
  if (ieta==58) fCal = 0.995338;
  if (ieta==59) fCal = 0.997791;
  if (ieta==60) fCal = 1.00248;
  if (ieta==61) fCal = 1.00579;
  if (ieta==62) fCal = 1.00427;
  if (ieta==63) fCal = 1.00537;
  if (ieta==64) fCal = 1.00672;
  if (ieta==65) fCal = 1.00257;
  if (ieta==66) fCal = 1.0049;
  if (ieta==67) fCal = 1.0046;
  if (ieta==68) fCal = 1.00795;
  if (ieta==69) fCal = 1.00395;
  if (ieta==70) fCal = 1.01101;
  if (ieta==71) fCal = 1.00737;
  if (ieta==72) fCal = 1.00423;
  if (ieta==73) fCal = 1.0072;
  if (ieta==74) fCal = 1.00661;
  if (ieta==75) fCal = 1.00912;
  if (ieta==76) fCal = 1.00598;
  if (ieta==77) fCal = 1.00861;
  if (ieta==78) fCal = 1.00715;
  if (ieta==79) fCal = 1.00848;
  if (ieta==80) fCal = 1.00633;
  if (ieta==81) fCal = 1.00914;
  if (ieta==82) fCal = 1.01473;
  if (ieta==83) fCal = 1.01143;
  if (ieta==84) fCal = 1.01382;
  if (ieta==85) fCal = 1.01723;
  if (ieta==86) fCal = 0.971282;
  if (ieta==87) fCal = 0.979535;
  if (ieta==88) fCal = 0.991313;
  if (ieta==89) fCal = 0.982041;
  if (ieta==90) fCal = 0.982812;
  if (ieta==91) fCal = 0.978579;
  if (ieta==92) fCal = 0.984946;
  if (ieta==93) fCal = 0.987141;
  if (ieta==94) fCal = 0.992083;
  if (ieta==95) fCal = 0.998825;
  if (ieta==96) fCal = 1.00955;
  if (ieta==97) fCal = 1.01208;
  if (ieta==98) fCal = 1.01213;
  if (ieta==99) fCal = 1.00755;
  if (ieta==100) fCal = 1.01093;
  if (ieta==101) fCal = 1.00523;
  if (ieta==102) fCal = 1.01429;
  if (ieta==103) fCal = 1.01619;
  if (ieta==104) fCal = 1.00907;
  if (ieta==105) fCal = 1.00662;
  if (ieta==106) fCal = 1.00313;
  if (ieta==107) fCal = 1.00408;
  if (ieta==108) fCal = 1.00763;
  if (ieta==109) fCal = 1.00456;
  if (ieta==110) fCal = 1.00113;
  if (ieta==111) fCal = 1.00423;
  if (ieta==112) fCal = 1.00402;
  if (ieta==113) fCal = 1.02568;
  if (ieta==114) fCal = 1.02646;
  if (ieta==115) fCal = 1.01995;
  if (ieta==116) fCal = 0.999479;
  if (ieta==117) fCal = 0.985867;
  if (ieta==118) fCal = 0.96131;
  return fCal;
}

void IC::applyEtaScale(IC & ic)
{
        TEndcapRings ring;
        for (size_t i = 0; i < ic.ids().size(); ++i) {
                DetId id(ic.ids()[i]);
                if (id.subdetId() == EcalBarrel) {
                        EBDetId eid(id);
                        ic.ic().setValue(id, fCalEoP(eid.ieta()));
                } else if (id.subdetId() == EcalEndcap) {
                        EEDetId eid(id);
                        ic.ic().setValue(id, fCalEoP(ring.GetEndcapIeta(eid.ix(), eid.iy(), eid.zside())));
                        //ic.ic().setValue(id, ring.GetEndcapIeta(eid.ix(), eid.iy(), eid.zside()));
                }
                ic.eic().setValue(id, 1);
        }
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

bool IC::isValid(float v, float e)
{
        //if (v < 0 || v > 2) return false;
        //if (v < 0) return false;
        if (fabs(e) > 100 || v < 0) return false;
        //if (fabs(e) > 100 || v < 0.4 || v > 2.5) return false;
        //if (v < 0.3 || v > 3) return false;
        return true;
}

void IC::scaleEta(IC & ic, const IC & ic_scale, bool reciprocalScale)
{
        float etasum[DRings::nHalfIEta * 2 + 1]; // ieta = 0 does not exist
        int n[DRings::nHalfIEta * 2 + 1]; // ieta = 0 does not exist
        for (int i = 0; i < DRings::nHalfIEta * 2 + 1; ++i) {
                etasum[i] = 0;
                n[i] = 0;
        }
        for (size_t i = 0; i < ic_scale.ids().size(); ++i) {
                DetId id(ic_scale.ids()[i]);
                float v = ic_scale.ic()[id];
                float e = ic_scale.eic()[id];
                if (!isValid(v, e)) continue;
                int idx = dr_.ieta(id) + DRings::nHalfIEta;
                etasum[idx] += v;
                n[idx]++;
        }
        for (size_t i = 0; i < ic.ids().size(); ++i) {
                DetId id(ic.ids()[i]);
                float v = ic.ic()[id];
                float e = ic.eic()[id];
                if (!isValid(v, e)) continue;
                int idx = dr_.ieta(id) + DRings::nHalfIEta;
                float s = etasum[idx] / n[idx];
                if (reciprocalScale) s = 1 / s;
                ic.ic().setValue(id, v * s);
                ic.eic().setValue(id, e * s);
        }
}


void IC::removeOutliers(const IC & a, IC & res, float low_thr, float high_thr)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];

                if (va > high_thr){
                        res.ic().setValue(id, high_thr);
                        fprintf(stderr, "[IC::RemoveOutlier] !!! set to 2.5 IC for crystal %d was %f now %f\n", id.rawId(), va, res.ic()[id]);
                }

                if (va < low_thr ){
                        res.ic().setValue(id, low_thr);
                        fprintf(stderr, "[IC::RemoveOutlier] !!! set to 0.4 IC for crystal %d was %f now %f\n", id.rawId(), va, res.ic()[id]);

                }
        }
}


void IC::dumpOutliers(const IC & a, float min, float max)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];

                IC::Coord c;
                coord(id, &c);

                if (va > max || va < min){
                        fprintf(stdout, "%d %d %d %d %f %f\n", id.rawId(), c.ix_, c.iy_, c.iz_, va, a.eic()[id]);
                }
        }
}


void IC::setToUnit(IC & ic)
{
        for (size_t i = 0; i < ic.ids().size(); ++i) {
                DetId id(ic.ids()[i]);
                ic.ic().setValue(id, 1.);
                ic.eic().setValue(id, 998);
        }
}


void IC::applyTwoCrystalEffect(IC & ic)
{
        for (size_t i = 0; i < ic.ids().size(); ++i) {
                DetId id(ic.ids()[i]);
                if (id.subdetId() == EcalBarrel) {
                        EBDetId eid(id);
                        float c = 1;
                        if (eid.zside() > 0) {
                                if (eid.iphi() % 2 == 1) c = 0.998;
                                if (eid.iphi() % 2 == 0) c = 1.002;
                        } else {
                                if (eid.iphi() % 2 == 0) c = 0.998;
                                if (eid.iphi() % 2 == 1) c = 1.002;
                        }
                        ic.ic().setValue(id, c * ic.ic()[id]);
                }
        }
}


void IC::fillHoles(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                if (isValid(va, ea)) {
                        res.ic().setValue(id, va);
                        res.eic().setValue(id, ea);
                } else if (isValid(vb, eb)) {
                        res.ic().setValue(id, vb);
                        res.eic().setValue(id, eb);
                } else {
                        fprintf(stderr, "[IC::fillHoles] WARNING: no IC for crystal %d\n", id.rawId());
                        res.ic().setValue(id, 1);
                        res.eic().setValue(id, 999);
                }
        }
}


void IC::readSimpleTextFile(const char * fileName, IC & ic)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readSimpleTextFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int id;
        float c, e;
        while ((read = getline(&line, &len, fd)) != EOF) {
                if (line[0] == '#') continue;
                sscanf(line, "%d %f %f", &id, &c, &e);
                ic.ic().setValue(id, c);
                ic.eic().setValue(id, e);
        }
        fclose(fd);
}


void IC::readTextFile(const char * fileName, IC & ic)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readTextFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int ix, iy, iz;
        float c, e;
        DetId id;
        while ((read = getline(&line, &len, fd)) != EOF) {
                if (line[0] == '#') continue;
                sscanf(line, "%d %d %d %f %f", &ix, &iy, &iz, &c, &e);
                if (iz == 0) id = EBDetId(ix, iy);
                else         id = EEDetId(ix, iy, iz);
                ic.ic().setValue(id, c);
                ic.eic().setValue(id, e);
        }
        fclose(fd);
}


void IC::readCmscondXMLFile(const char * fileName, IC & ic)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readXMLFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int n, nvalue = 0, nxtals = 0, cnt = 0;
        float c;
        char cc;
        DetId id;
        while ((read = getline(&line, &len, fd)) != EOF) {
                if ((n = sscanf(line, " </Array%c", &cc)) == 1) {
                        assert(cc == '>');
                        assert(cnt == 61200 || cnt == 14648);
                        nvalue = 0;
                        cnt = 0;
                }
                if (nvalue && (n = sscanf(line, " <Float_t v=\"%f", &c)) == 1) {
                        if      (nxtals == 61200) id = EBDetId::unhashIndex(cnt);
                        else if (nxtals == 14648) id = EEDetId::unhashIndex(cnt);
                        printf("--> %u %d %f\n", id.rawId(), cnt, c);
                        ic.ic().setValue(id, c);
                        ic.eic().setValue(id, 0);
                        ++cnt;
                } else if (nvalue) {
                        fprintf(stderr, "[readXMLFile] problem reading line `%s'", line);
                }
                if ((n = sscanf(line, " <Int_t v=\"%d\"", &nxtals)) == 1) {
                        getline(&line, &len, fd);
                        nvalue = 1;
                } else if ((n = sscanf(line, " <Int_t v=\"%d\"", &nxtals)) == 1) {
                        getline(&line, &len, fd);
                        nvalue = 1;
                }
        }
}


void IC::readXMLFile(const char * fileName, IC & ic)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readXMLFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int ix, iy, iz;
        int n, nvalue = 0;
        float c;
        DetId id;
        while ((read = getline(&line, &len, fd)) != EOF) {
                if (nvalue && (n = sscanf(line, " <Value>%f", &c)) == 1) {
                        //printf("ix= %d iy= %d iz= %d --> %f\n", ix, iy, iz, c);
                        if (iz == 0) id = EBDetId(ix, iy);
                        else         id = EEDetId(ix, iy, iz);
                        ic.ic().setValue(id, c);
                        ic.eic().setValue(id, 0);
                        nvalue = 0;
                } else if (nvalue) {
                        fprintf(stderr, "[readXMLFile] problem reading line `%s'", line);
                }
                if ((n = sscanf(line, " <cell iEta=\"%d\"", &ix)) == 1) {
                        sscanf(line, " <cell iEta=\"%d\" iPhi=\"%d\"", &ix, &iy);
                        iz = 0;
                        nvalue = 1;
                } else if ((n = sscanf(line, " <cell ix=\"%d\"", &ix)) == 1) {
                        sscanf(line, " <cell ix=\"%d\" iy=\"%d\" zside=\"%d\"", &ix, &iy, &iz);
                        nvalue = 1;
                }
        }
}


void IC::readEcalChannelStatusFromTextFile(const char * fileName, EcalChannelStatus & channelStatus)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readSimpleTextFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        uint32_t id;
        uint16_t st;
        int cnt = 0;
        while (fscanf(fd, "%u %hu", &id, &st) != EOF) {
                EcalChannelStatusCode code(st);
                channelStatus[id] = code;
                ++cnt;
        }
        assert(cnt == 75848);
        fclose(fd);
}


void IC::makeRootTree(TTree & t, const IC & ic)
{
}
