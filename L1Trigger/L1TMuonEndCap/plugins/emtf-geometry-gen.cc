#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include <L1Trigger/L1TMuonEndCap/plugins/emtf-geometry-gen.h>

#define UPPER_THETA 45
#define LOWER_THETA 8.5



//the boolean isTMB07 is set to true, but I still don't know what it's for.

using namespace std;

slhc_geometry::slhc_geometry(edm::ParameterSet const& conf)
{
	// lutParam is necessary to make an instance of CSCSectorReceiverLUT (which is not done at the moment)
  	lutParam = conf.getParameter<edm::ParameterSet>("lutParam");
}

//this function is executed first in every run
void slhc_geometry::beginJob(edm::EventSetup const& es)
{

}

//this function is executed last in every run
void slhc_geometry::endJob()
{

}


void slhc_geometry::analyze(edm::Event const& e, edm::EventSetup const& es)
{
    edm::ESHandle<CSCGeometry> pDD;
    es.get<MuonGeometryRecord>().get( pDD );
    CSCTriggerGeometry::setGeometry(pDD);

    int half_strip[4] = {29,29,129,128};
    unsigned ring, ichamber;



    // create ME11 image for geometry check
    std::ostringstream me11_img_name;
    me11_img_name << "ME11_e1_ch1_image.csv";
    std::ofstream me11_img;
    me11_img.open(me11_img_name.str().c_str());

    float me11_ph, me11_th;
    get_ring_chamber(1, 1, 1, 1, ring, ichamber);
    cout << "cscid = " << 1 << " ring " << ring << " chamber " << ichamber << endl;
    int me11_ph_int, me11_th_int;

    for (int w = 0; w < 48; w++) // wire loop
    {
        for (int s = 0; s < 48; s++) // strip loop
        {
            me11_ph = get_sector_phi_hs(/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/0, /*strip*/s*2, /*cscId*/1, false, es);
            me11_th = getTheta         (/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/w, /*strip*/s,   /*cscId*/1, es);
//            me11_th_lim = getTheta_limited         (/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/w, /*strip*/s,   /*cscId*/1, es);

//            if (w == 0) me11_img << "w = 0 s = " << s << " th: " << me11_th << endl;

            me11_th_int = (me11_th - LOWER_THETA)*128/(UPPER_THETA - LOWER_THETA);
            me11_ph_int = (int)roundl((me11_ph) / (10. / 75. / 8.));

            me11_img << s << ", " << w << ", " << me11_ph_int << ", " << me11_th_int << endl;
        }

    }

    me11_img.close();

    // create ME11 image for geometry check
    std::ostringstream me11_img_name1;
    me11_img_name1 << "ME11_e1_ch10_image.csv";
    me11_img.open(me11_img_name1.str().c_str());
    get_ring_chamber(1, 1, 1, 10, ring, ichamber);
    cout << "cscid = " << 10 << " ring " << ring << " chamber " << ichamber << endl;
    get_ring_chamber(1, 1, 1, 12, ring, ichamber);
    cout << "cscid = " << 12 << " ring " << ring << " chamber " << ichamber << endl;
    get_ring_chamber(1, 1, 1, 13, ring, ichamber);
    cout << "cscid = " << 13 << " ring " << ring << " chamber " << ichamber << endl;

    for (int w = 0; w < 48; w++) // wire loop
    {
        for (int s = 0; s < 48; s++) // strip loop
        {
            me11_ph = get_sector_phi_hs(/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/0, /*strip*/s*2, /*cscId*/10, false, es);
            me11_th = getTheta         (/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/w, /*strip*/s,   /*cscId*/10, es);

 //           if (w == 0) me11_img << "w = 0 s = " << s << " th: " << me11_th << endl;

            me11_th_int = (me11_th - LOWER_THETA)*128/(UPPER_THETA - LOWER_THETA);
            me11_ph_int = (int)roundl((me11_ph) / (10. / 75. / 8.));

            me11_img << s << ", " << w << ", " << me11_ph_int << ", " << me11_th_int << endl;
        }

    }

    me11_img.close();

    // create ME11 image for geometry check
    std::ostringstream me11_img_name2;
    me11_img_name2 << "ME11_e2_ch1_image.csv";
    me11_img.open(me11_img_name2.str().c_str());
    get_ring_chamber(1, 1, 1, 1, ring, ichamber);
    cout << "cscid = " << 1 << " ring " << ring << " chamber " << ichamber << endl;

    for (int w = 0; w < 48; w++) // wire loop
    {
        for (int s = 0; s < 48; s++) // strip loop
        {
            me11_ph = get_sector_phi_hs(/*endcap*/2, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/0, /*strip*/s*2, /*cscId*/1, false, es);
            me11_th = getTheta         (/*endcap*/2, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/w, /*strip*/s,   /*cscId*/1, es);

            me11_th_int = (me11_th - LOWER_THETA)*128/(UPPER_THETA - LOWER_THETA);
            me11_ph_int = (int)roundl((me11_ph) / (10. / 75. / 8.));

            me11_img << s << ", " << w << ", " << me11_ph_int << ", " << me11_th_int << endl;
        }

    }

    me11_img.close();

    // create ME11 image for geometry check
    std::ostringstream me11_img_name3;
    me11_img_name3 << "ME11_e1_ch2_image.csv";
    me11_img.open(me11_img_name3.str().c_str());

    for (int w = 0; w < 48; w++) // wire loop
    {
        for (int s = 0; s < 48; s++) // strip loop
        {
            me11_ph = get_sector_phi_hs(/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/0, /*strip*/s*2, /*cscId*/2, false, es);
            me11_th = getTheta         (/*endcap*/1, /*station*/1, /*sector*/1, /*subsector*/1, /*wireGroup*/w, /*strip*/s,   /*cscId*/2, es);

            me11_th_int = (me11_th - LOWER_THETA)*128/(UPPER_THETA - LOWER_THETA);
            me11_ph_int = (int)roundl((me11_ph) / (10. / 75. / 8.));

            me11_img << s << ", " << w << ", " << me11_ph_int << ", " << me11_th_int << endl;
        }

    }

    me11_img.close();
    double phi[4] = {0,0,0,0};
    phi[0] = get_sector_phi_hs(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/half_strip[0],  /*cscId*/4, false, es);
    phi[1] = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/half_strip[1],  /*cscId*/4, false, es);
    phi[2] = get_sector_phi_hs(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/half_strip[2],  /*cscId*/4, false, es);
    phi[3] = get_sector_phi_hs(/*endcap*/1,  /*station*/4,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/half_strip[3],  /*cscId*/4, false, es);

    cout << "phi: " << phi[0] << " " << phi[1] << " " << phi[2] << " " << phi[3] << " " << endl;

    int phi_int[4];
    for (int i = 0; i < 4; i++)
    {
        phi_int[i] = (int)roundl((phi[i]) / (10. / 75. / 8.));
    }

    cout << "phi_int: " << phi_int[0] << " " << phi_int[1] << " " << phi_int[2] << " " << phi_int[3] << " " << endl;

    phi[0] = get_sector_phi_hs(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/0,  /*cscId*/4, false, es);
    phi[1] = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/0,  /*cscId*/4, false, es);
    phi[2] = get_sector_phi_hs(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/159,  /*cscId*/4, false, es);
    phi[3] = get_sector_phi_hs(/*endcap*/1,  /*station*/4,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/159,  /*cscId*/4, false, es);

    cout << "phi: " << phi[0] << " " << phi[1] << " " << phi[2] << " " << phi[3] << " " << endl;

    for (int i = 0; i < 4; i++)
    {
        phi_int[i] = (int)roundl((phi[i]) / (0.1333333 / 8.));
    }


    cout << "phi_int: " << phi_int[0] << " " << phi_int[1] << " " << phi_int[2] << " " << phi_int[3] << " " << endl;

    phi[0] = get_sector_phi_hs(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/159,  /*cscId*/4, false, es);
    phi[1] = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/159,  /*cscId*/4, false, es);
    phi[2] = get_sector_phi_hs(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/0,  /*cscId*/4, false, es);
    phi[3] = get_sector_phi_hs(/*endcap*/1,  /*station*/4,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/0,  /*cscId*/4, false, es);

    cout << "phi: " << phi[0] << " " << phi[1] << " " << phi[2] << " " << phi[3] << " " << endl;

    for (int i = 0; i < 4; i++)
    {
        phi_int[i] = (int)roundl((phi[i]) / (0.1333333 / 8.));
    }


    cout << "phi_int: " << phi_int[0] << " " << phi_int[1] << " " << phi_int[2] << " " << phi_int[3] << " " << endl;

    for (int i = 0; i < 159; i++)
    {
        phi[1] = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/i,  /*cscId*/4, false, es);
        phi_int[1] = (int)roundl((phi[1]) / (10. / 75. / 8.));
        cout << i << " " << phi[1] << " " << phi_int[1] << endl;
    }


    // ME11 scan
    for (int i = 0; i < 48; i++)
    {
        phi[0] = getGlobalPhiValue(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/i,  /*strip*/0,  /*cscId*/1, es);
        phi[1] = getGlobalPhiValue(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/i,  /*strip*/47,  /*cscId*/1, es);
        phi[2] = getGlobalPhiValue(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/i,  /*strip*/0,  /*cscId*/10, es);
        phi[3] = getGlobalPhiValue(/*endcap*/1,  /*station*/1,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/i,  /*strip*/47,  /*cscId*/10, es);
        cout << "wg: " << i << " phi at strip 0: " << phi[0] << " phi at strip 89: " << phi[1] << " phi at strip 128: " << phi[2] << " phi at strip 128+89: " << phi[3] << endl;
    }

    // sector scan
    for (int si = 1; si <= 6; si++) // sector
    {
        for (int st = 1; st <= 4; st++) // station
        {
            int istrip = 0;
            if (st > 2) istrip = 79;
            phi[0] = getGlobalPhiValue(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/1, es);
            phi[1] = getGlobalPhiValue(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/4, es);
            phi[2] = getGlobalPhiValue(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/7, es);
            phi[3] = getGlobalPhiValue(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/10,es); // me11a
            cout << "sector: " << si << " station: " << st << " phi: " << setw(6) << phi[0] << " " << setw(6) << phi[1] << " " << setw(6) << phi[2] << " " << setw(6) << phi[3] << endl;
        }
    }

    // sector scan
    for (int si = 1; si <= 6; si++) // sector
    {
        for (int st = 1; st <= 4; st++) // station
        {
            int istrip = 0;
            if (st > 2) istrip = 159;
            phi[0] = get_sector_phi_hs(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/1, false, es);
            phi[1] = get_sector_phi_hs(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/4, false, es);
            phi[2] = get_sector_phi_hs(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/7, false, es);
            phi[3] = get_sector_phi_hs(/*endcap*/1,  /*station*/st,  /*sector*/si,  /*subsector*/1,  /*wireGroup*/0,  /*strip*/istrip,  /*cscId*/10, false,es); // me11a
            cout << "sector: " << si << " station: " << st << " phi: " << setw(6) << phi[0] << " " << setw(6) << phi[1] << " " << setw(6) << phi[2] << " " << setw(6) << phi[3] << endl;
        }
    }


    // uncomment this section to generate chamber origin LUTs

    for (int endcap = 1; endcap <= 2; endcap++)
    {
        int num_of_wires[5][13];
        memset(num_of_wires, 0, sizeof (num_of_wires));
        memset(num_of_wiregroups, 0, sizeof (num_of_wiregroups));
        memset(num_of_strips, 0, sizeof (num_of_strips));
        memset(strip_phi_pitch, 0, sizeof (strip_phi_pitch));
        memset(strip_dphi, 0, sizeof (strip_dphi));


        for (int sect = 1; sect <= 6; sect++)
        {
            for (int station = 1; station <= 4; station ++)
            {
                for (int cscId = 1; cscId <= 12; cscId++)
                {
                    // ME1/1A only in ME1 station
                    if (station > 1 && cscId > 9) continue;

                    unsigned ring = 0, ichamber = 0;

                    get_ring_chamber(station, sect, 1, cscId, ring, ichamber);

                    CSCDetId id = CSCDetId(endcap, station, ring, ichamber, CSCConstants::KEY_CLCT_LAYER);
                    edm::ESHandle<CSCGeometry> cscGeom;
                    es.get<MuonGeometryRecord>().get(cscGeom);

                    const CSCChamber* chamber = cscGeom->chamber(id);
                    CSCLayerGeometry* layerGeom = const_cast<CSCLayerGeometry*>(chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());

                    LocalPoint lPoint1 = layerGeom->stripWireGroupIntersection(1, 1); // strip and wg in geometry routines start from 1
                    GlobalPoint gPoint1 = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint1);

                    LocalPoint lPoint2 = layerGeom->stripWireGroupIntersection(2, 1); // strip and wg in geometry routines start from 1
                    GlobalPoint gPoint2 = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint2);

                    double phi1 = gPoint1.phi();
                    double phi2 = gPoint2.phi();
                    double dphi = fabs(phi2 - phi1);

                    // get number of wires (not wiregroups)
                    num_of_wires[station][cscId] = layerGeom->numberOfWires();
                    num_of_wiregroups[sect-1][station][cscId] = layerGeom->numberOfWireGroups();
                    num_of_strips    [sect-1][station][cscId] = layerGeom->numberOfStrips();
                    strip_phi_pitch  [sect-1][station][cscId] = layerGeom->stripPhiPitch();
                    strip_dphi       [sect-1][station][cscId] = dphi;
                }
            }
        }
        generateLUTStation1(endcap, es);
        generateLUTs(endcap, es);
    }


    /*
    // the code below generates theta coverage map

    for (int station = 1; station <= 4; station ++)
        for (int cscId = 1; cscId <= 9; cscId++)
            std::cout << "st " << station << " ch " << cscId << " wires " << num_of_wires[station][cscId] << std::endl;

    std::cout << std::endl;


    // generating theta coverage map
    // using first and last wires in each chamber (NOT wiregroups)
    for (int csc = 1; csc <= 7; csc += 3)
    {
        for (int wg = 0; wg < num_of_wires[1][csc]; wg += num_of_wires[1][csc]-1)
        {
            int strip = (wg == 0) ? 40 : 0; // midlle of chamber at bottom, corner at top, to maximize coverage
            if (csc == 1 && wg == 0) strip = 0; // tilted wires, take bottom
            if (csc == 1 && wg != 0) strip = 79; // tilted wires, take top
            if (csc == 7) strip = (wg == 0) ? 32 : 0; // ME1/3 middle strip
            double th = getTheta_wire
            (
                1, // endcap
                1, // station
                1, // sector
                1, // subsector
                wg, // wireGroup
                strip, // strip
                csc // cscId
            );
            std::cout << "st 1 csc " << csc << " wg " << wg << " th " << th << std::endl;
        }
    }

    for (int st = 2; st <= 4; st++)
    {
        for (int csc = 1; csc <= 4; csc += 3)
        {
            if (st == 4 && csc == 4) break; // ME4/2 not implemented in CMSSW yet

            for (int wg = 0; wg < num_of_wires[st][csc]; wg += num_of_wires[st][csc]-1)
            {
                double th = getTheta_wire
                (
                    1, // endcap
                    st, // station
                    1, // sector
                    1, // subsector
                    wg, // wireGroup
                    (wg == 0) ? 40 : 0, // midlle of chamber at bottom, corner at top, to maximize coverage
                    csc // cscId
                );
                std::cout << "st " << st << " csc " << csc << " wg " << wg << " th " << th << std::endl;
            }
        }
    }
*/
    // end of theta coverage map code
}


// generates correction LUTs for station 1 where the wires are tilted. prints theta differences (in bits) in files (one file for each chamber). 
// for each chamber a file is created that contains the theta values (in degrees) for strip 0 for all wiregroups.
void slhc_geometry::generateLUTStation1(int endcap, edm::EventSetup const& es) 
{

    std::ofstream thetaDiffStream;
    std::stringstream ss2;
    // insert endcap into file name!!!
    ss2 << "theta_corr_endcap_" << endcap << ".cpp";
    std::string outFile2(ss2.str());
    thetaDiffStream.open(outFile2.c_str());

    int station = 1;

    int real_cham[] = {1,2,3,1}; // real cscids
    int file_cham[] = {1,2,3,13}; // chamber # for file name
    int real_sect[] = {6,1,2,3,4,5}; // neighbor sector

    for (int seci = 1; seci <= 6; seci++)
    {
        for (int subi = 1; subi <= 2; subi++)
        {
            int tot_cham = (subi == 1 ? 4 : 3);
            for (int chami = 1; chami <= tot_cham; chami++)
            {

                int sector;
                int chamber = real_cham[chami-1];
                int subsector;
                if (subi == 1 && chami == 4) // neighbor sector chamber
                {
                    sector = real_sect[seci-1];
                    subsector = 2;
                }
                else
                {
                    sector = seci;
                    subsector = subi;
                }

                int maxWire = 48;
                thetaDiffStream << "void th_corr_lut_" << seci << "_" << subi << "_" << station << "_" << file_cham[chami-1] << "(signal_ &index, signal_ &th_corr)\n{\n  begincase(index)\n";

                std::stringstream fn;
                fn << "vl_th_corr_lut_endcap_" << endcap << "_sec_" <<  seci << "_sub_" << subi << "_st_" << station << "_ch_" << file_cham[chami-1] << ".lut";
                std::ofstream lutfile; // open separate file for each lut, for verilog test fixture
                lutfile.open(fn.str().c_str());
                fn.str("");

                // select correction points at 1/6, 3/6 and 5/6 of chamber wg range
                // this makes construction of LUT address in firmware much easier
                int index = 0;
                for (int wire = maxWire/6; wire < maxWire; wire += maxWire/3)
                {

                    thetaDiffStream << "    // wire = " << wire << std::endl;

                    // the 0 below is strip 0
                    double thetaStrip0 = getTheta(endcap, station, sector, subsector, wire, 0, chamber, es);
                    for (int strip = 0; strip < 64; strip+=2) // pattern search works in double strips, so take every other strip
                    {
                        int diff = abs((int)roundl(128*(getTheta(endcap, station, sector, subsector, wire, strip, chamber, es) -
                                                        thetaStrip0)/(UPPER_THETA - LOWER_THETA)));
                        // diff may be negative for chambers in negative endcap, because wire tilt is the opposite way
                        // firmware will subtract this correction instead of adding it
                        thetaDiffStream << "    case1(" << setw(3) << index << ") th_corr = " << setw(2) << diff << ";" << std::endl;
                        lutfile << hex << diff << std::endl;
                        index++;
                    }

                }

                thetaDiffStream << "  endcase\n};\n";
                lutfile.close();
            }
        }
    }

    thetaDiffStream.close();

}



int ph_coverage_max[5][3], th_coverage_max[5][3];

//Generates LUTs for all chambers in one endcap at the time (change the variable endcap for endcap 2). Phi values are shot at strip 0 and wiregroup 0 for every chamber.
//Theta values are recorded for strip 0 and the middle strip for all wiregroups in all chambers.
void slhc_geometry::generateLUTs(int endcap, edm::EventSetup const& es) 
{

	std::ostringstream phi_stream_name;
	phi_stream_name << "phiValuesStripZero_endcap_"<< endcap <<".txt";
	std::ofstream phiStream;
	phiStream.open(phi_stream_name.str().c_str());
	//bitValue = k*phiValue + m
	double k = 128/(UPPER_THETA - LOWER_THETA);
	double m = -k*LOWER_THETA;

    int th_init[6][5][16];
	memset (th_init, 0, sizeof (th_init));

    int th_cover[6][5][16];
	memset (th_cover, 0, sizeof (th_cover));

    int ph_init[6][5][16];
	memset (ph_init, 0, sizeof (ph_init));

    int ph_init_full[6][5][16];
	memset (ph_init_full, 0, sizeof (ph_init_full));

    int ph_cover[6][5][16];
	memset (ph_cover, 0, sizeof (ph_cover));

    bool ph_reverse[6][5][16];
	memset (ph_reverse, 0, sizeof (ph_reverse));

	//for the middle strip
	std::stringstream ss_4th;
	ss_4th << "ME_strip4th_endcap_"<< endcap <<".cpp";
	std::string outFile_4th(ss_4th.str());
	std::ofstream thetaStream_4th;
	thetaStream_4th.open(outFile_4th.c_str());

    cout << "ME1/1A strip pitch: " << setw(12) << (strip_phi_pitch[0][1][1]*180./M_PI) << " measured "  << setw(12) << (strip_dphi[0][1][1]*180./M_PI) << endl;
    cout << "ME1/1B strip pitch: " << setw(12) << (strip_phi_pitch[0][1][10]*180./M_PI) << "measured  "  << setw(12) << (strip_dphi[0][1][10]*180./M_PI) <<  endl;
    cout << "ME2/2 strip pitch: " << setw(12) << (strip_phi_pitch[0][2][4]*180./M_PI) << " measured "  << setw(12) << (strip_dphi[0][2][4]*180./M_PI) <<  endl;
	
    double g_s2c1s0 = getGlobalPhiValue(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/0,  /*cscId*/1, es);
    double g_s3c1s0 = getGlobalPhiValue(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/0,  /*cscId*/1, es);
    double g_s2c1sM = getGlobalPhiValue(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/79,  /*cscId*/1, es);
    double g_s3c1sM = getGlobalPhiValue(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/79,  /*cscId*/1, es);

	cout << setw(12) << g_s2c1s0 << " " << setw(12) << g_s3c1s0 << " " << setw(12) << g_s2c1sM << " " << setw(12) << g_s3c1sM << endl;

    double l_s2c1s0 = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/0,  /*cscId*/1, false, es);
    double l_s3c1s0 = get_sector_phi_hs(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/0,  /*cscId*/1, false, es);
    double l_s2c1sM = get_sector_phi_hs(/*endcap*/1,  /*station*/2,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/79,  /*cscId*/1, false, es);
    double l_s3c1sM = get_sector_phi_hs(/*endcap*/1,  /*station*/3,  /*sector*/1,  /*subsector*/1,  /*wireGroup*/60,  /*strip*/79,  /*cscId*/1, false, es);

	cout << setw(12) << l_s2c1s0 << " " << setw(12) << l_s3c1s0 << " " << setw(12) << l_s2c1sM << " " << setw(12) << l_s3c1sM << endl;


	for (int station = 1; station<=4; station++) 
	{
		if (station == 1) 
		{
			for (int sector = 1; sector <= 6; sector++) 
			{
				for (int subsector = 1; subsector <= 2; subsector++) 
				{
					phiStream << "station " << station << ", sector " << sector << ", subsector " << subsector << std::endl;
                    for (int chamber = 1; chamber <= ((subsector == 1) ? 16 : 12); chamber++) // subsector 1 includes neighbor chambers
					{
					
						
						thetaStream_4th << "void th_lut_" << sector << "_" << subsector << "_" << station << "_" << chamber << "(signal_ &wg, signal_ &th)\n{\n  begincase(wg)\n";

						std::stringstream fn;
						fn << "vl_th_lut_endcap_" << endcap << "_sec_" <<  sector << "_sub_" << subsector << "_st_" << station << "_ch_" << chamber << ".lut";
						std::ofstream lutfile; // open separate file for each lut, for verilog test fixture
						lutfile.open(fn.str().c_str());
						fn.str("");
						
                        int rcscid = (chamber < 13) ? chamber : (chamber == 13) ? 3 : (chamber == 14) ? 6 : (chamber == 15) ? 9 : 12; // real cscid number
                        int rsubsector = (chamber < 13) ? subsector : 2; // real subsector (neighbor sector's chambers are in its subsector 2)
                        unsigned rsector; // real sector
                        if (chamber >= 13)
                        {
                            if (sector > 1) rsector = sector - 1; else rsector = 6; // find neighbor sector
                        }
                        else
                            rsector = sector;

                        int maxWire;
                        if (rcscid < 4 || rcscid > 9)
							maxWire = 48;
                        else if (rcscid < 7)
							maxWire = 64;
						else 
							maxWire = 32;

                        // wiregroup and halfstrip are set to zero, respectively
                        double fphi_first = get_sector_phi_hs(endcap, station, rsector, rsubsector, 0, 0, rcscid, chamber >= 13, es);
                        double fphi_last  = get_sector_phi_hs(endcap, station, rsector, rsubsector, 0,
                                              num_of_strips[rsector-1][station][rcscid]*2-1, rcscid, chamber >= 13, es);
                        if (fphi_first > fphi_last) ph_reverse[sector-1][subsector-1][chamber-1] = true;

						double fphi_diff = fabsl(fphi_last - fphi_first)/2.; // in double-strips

						int phiint =  (int)roundl(fphi_first/(strip_phi_pitch[0][2][4]*180./M_PI));
						//double pitch18 = strip_phi_pitch[0][2][4] / 8; // 1/8 strip pitch, for full precision phi
						//int phiint_full =  (int)roundl(fphi_first/(pitch18*180./M_PI));
						int phiint_full =  (int)roundl(fphi_first/(10./75./8.));

						ph_init_full[sector-1][subsector-1][chamber-1] = phiint_full;
						ph_init[sector-1][subsector-1][chamber-1] = phiint;
						ph_cover[sector-1][subsector-1][chamber-1] = (int)roundl(fphi_diff/(strip_phi_pitch[0][2][4]*180./M_PI));
						phiStream << setw(3) << phiint << "   " << setw(10) << fphi_first << "  " << setw(10) << fphi_last << std::endl;

						int maxStrip;
                        if (rcscid > 9)
						  maxStrip = 48; // ME1/1A
                        else if (rcscid >= 7 && rcscid <= 9)
						  maxStrip = 64;
						else
						  maxStrip = 80;

						// find th_init and th_cover for ME1/1. Need to take opposite corners of chamber according to wire tilt, 
						// also keep in mind that + and - endcaps are different
						int bot_str, top_str;
						double fth_init;

                        if (rcscid <= 3 || rcscid > 9) // ME 1/1
						{
							// select top and bottom strip according to endcap
							// basically, need to hit the corners of the chamber with truncated wires (relevant for ME1/1 only)
//							if (endcap == 1) {top_str = 0; bot_str = maxStrip-1;}
//							else             {bot_str = 0; top_str = maxStrip-1;}
                            if (endcap == 1) {top_str = 47; bot_str = 0;}
                            else             {bot_str = 47; top_str = 0;}
                        }
						else 
						{
							bot_str = top_str = maxStrip/4;
						}

						// find theta at top and bottom of chamber
                        fth_init = getTheta(endcap, station, rsector, rsubsector, 0, bot_str, rcscid, es);

						// widen ME1/1 coverage slightly, because of odd geometry of truncated wiregroups
//                        if (rcscid <= 3 || rcscid > 9)
//						{
//							fth_init -= 1/k; // this is one click in theta scale (not in degrees)
//						}

						th_init[sector-1][subsector-1][chamber-1] = (int)roundl(k*fth_init +m);
						th_cover[sector-1][subsector-1][chamber-1] = 
                          (int)roundl(k*(getTheta(endcap, station, rsector, rsubsector, maxWire-1, top_str, rcscid, es) -fth_init));

						// widen ME1/1 coverage slightly, because of odd geometry of truncated wiregroups
                        if (rcscid <= 3 || rcscid > 9)
						{
							th_cover[sector-1][subsector-1][chamber-1] += 2;
						}

						for (int wire = 0; wire < maxWire; wire++) 
						{
                            int hitstrip = (rcscid > 3 && rcscid <= 9) ? maxStrip/4 : 0;
							int bitValue = 
                              (int)roundl(k*(getTheta(endcap, station, rsector, rsubsector, wire, hitstrip, rcscid, es) - fth_init));

							// apply chamber coverage start
//							bitValue -= th_init[sector-1][subsector-1][chamber-1];

							// bitValue can become negative for truncated tilted wires in ME1/1
							// convert negative values into a large number so firmware will cut it off (relevant for ME1/1 only)
							if (bitValue < 0) bitValue &= 0x3f;
							// take 1/4 of max strip to minimize displacement due to straight wires in polar coordinates (all chambers except ME1/1)
							thetaStream_4th << "    case1(" << wire << ") th = " << bitValue << ";" << std::endl;
							lutfile << hex << bitValue << std::endl;
						}

								
						thetaStream_4th << "  endcase\n};\n";
						lutfile.close();
					}
					phiStream << std::endl;
				}
			}
		}

		else if (station == 2 || station == 3 ||  station == 4)
		{
			for (int sector = 1; sector <= 6; sector++)
			{
				phiStream << "station " << station << ", sector " << sector << std::endl;
                for (int chamber = 1; chamber <= 11; chamber++) // chambers 10 and 11 are from neighbor sector, 3 and 9 respectively
				{
					
					thetaStream_4th << "void th_lut_" << sector << "_" << station << "_" << chamber << "(signal_ &wg, signal_ &th)\n{\n  begincase(wg)\n";

					std::stringstream fn;
					fn << "vl_th_lut_endcap_" << endcap << "_sec_" <<  sector << "_st_" << station << "_ch_" << chamber << ".lut";
					std::ofstream lutfile; // open separate file for each lut, for verilog test fixture
					lutfile.open(fn.str().c_str());
					fn.str("");
					
                    int maxWire;
                    int rcscid = (chamber < 10) ? chamber : (chamber == 10) ? 3 : 9; // real cscid number
                    unsigned rsector; // real sector
                    if (chamber >= 10)
                    {
                        if (sector > 1) rsector = sector - 1; else rsector = 6; // find neighbor sector
                    }
                    else
                        rsector = sector;


                    if (rcscid < 4 && station == 2)
						maxWire = 112;
                    else if (rcscid >= 4 && station == 2)
						maxWire = 64;
                    else if (rcscid < 4 && station >= 3)
						maxWire = 96;
                    else if (rcscid >= 4 && station >= 3)
						maxWire = 64;
					else
						maxWire = 0;
					
					//the subsector is set to one whereas wiregroup and halfstrip are set to zero, respectively
                    double fphi_first = get_sector_phi_hs(endcap, station, rsector, 1, 0, 0, rcscid, chamber >= 10, es);
                    double fphi_last  = get_sector_phi_hs(endcap, station, rsector, 1, 0,
                                          num_of_strips[rsector-1][station][rcscid]*2-1, rcscid, chamber >= 10, es);
                    if (fphi_first > fphi_last) ph_reverse[sector-1][station][chamber-1] = true;

					double fphi_diff = fabsl(fphi_last - fphi_first)/2.; // in double strips

					int phiint =  (int)roundl(fphi_first/(strip_phi_pitch[0][2][4]*180./M_PI));
					//double pitch18 = strip_phi_pitch[0][2][4] / 8; // 1/8 strip pitch, for full precision phi
					//int phiint_full =  (int)roundl(fphi_first/(pitch18*180./M_PI));
					int phiint_full =  (int)roundl(fphi_first/(10./75./8.));
		
                    ph_init_full[sector-1][station][chamber-1] = phiint_full;
                    ph_init[sector-1][station][chamber-1] = phiint;
                    ph_cover[sector-1][station][chamber-1] = (int)roundl(fphi_diff/(strip_phi_pitch[0][2][4]*180./M_PI));
					phiStream << setw(3) << phiint << "   " << setw(10) << fphi_first << "  " << setw(10) << fphi_last << std::endl;
						
					for (int wire = 0; wire < maxWire; wire++) 
					{
						
						int maxStrip = 80;
												
						// take 1/4 of max strip to minimize displacement due to straight wires in polar coordinates
                        int bitValue = (int)roundl(k*getTheta(endcap, station, rsector, 1, wire, maxStrip/4, rcscid, es) +m);

                        double fth_rel = getTheta(endcap, station, rsector, 1, wire, maxStrip/4, rcscid, es) -
                          getTheta(endcap, station, rsector, 1, 0, maxStrip/4, rcscid, es);
						int th_rel = (int)(roundl(k * fth_rel));

						if (wire == 0) 
						{
                            th_init[sector-1][station][chamber-1] = bitValue;
                            thetaStream_4th << "// th_init = " << th_init[sector-1][station][chamber-1] << std::endl;
						}
						if (wire == maxWire-1)
						{
                            th_cover[sector-1][station][chamber-1] = th_rel;
                            thetaStream_4th << "// th_cover = " << th_cover[sector-1][station][chamber-1] << std::endl;
						}
						thetaStream_4th << "    case1(" << wire << ") th = " << th_rel << ";" << std::endl;
						lutfile << hex << th_rel << std::endl;
					}
					thetaStream_4th << "  endcase\n};\n";
					lutfile.close();
				}
				phiStream << std::endl;
			}
		}
	}

	thetaStream_4th.close();
	phiStream.close();

	// max coverage for chamber types (not individual chambers)
	// types are: 
	//	st1:   cscid 1-3, 4-6, 7-9
	// 	st234: cscid 1-3, 4-9

	if (endcap == 1) // reset max coverages only in the very beginning
	{
		memset (ph_coverage_max, 0, sizeof (ph_coverage_max));
		memset (th_coverage_max, 0, sizeof (th_coverage_max));
	}

	// update max coverages. Need to have max coverages that include both endcaps
	for (int sect = 0; sect < 6; sect++)
	{
		for (int s = 0; s < 5; s++)
		{
			for (int c = 0; c < 9; c++)
			{
				int ch_type = c / 3;
				if (s > 1 && ch_type > 1) ch_type = 1; // st 234 have only 2 chamber types

				if (ph_coverage_max[s][ch_type] < ph_cover[sect][s][c]) ph_coverage_max[s][ch_type] = ph_cover[sect][s][c];
				if (th_coverage_max[s][ch_type] < th_cover[sect][s][c]) th_coverage_max[s][ch_type] = th_cover[sect][s][c];
			}
		}
	}

	// values for ph and th init values hardcoded in verilog zones.v
	// these are with offset relative to actual init values to allow for chamber displacement
	// [station][chamber]
    // ME1 chambers 13,14,15,16 are neihbor sector's 3,6,9,12
    // ME2 chambers 10,11 are neighbor sector's 3,9
    int ph_init_hard[5][16] =
//	{
//	  {  4, 40, 78,  4, 42, 78,  8, 46, 84,  4, 40, 78},
//	  {116,154,190,116,154,192,122,158,196,116,154,190},
//	  {  2, 78,152,  4, 42, 78,116,154,190,  0,  0,  0},
//	  {  2, 78,152,  4, 42, 78,116,154,190,  0,  0,  0},
//	  {  2, 78,152,  2, 40, 78,116,152,190,  0,  0,  0}
//	};
    {
        {39,  57,  76, 39,  58,  76, 41,  60,  79, 39,  57,  76,21,21,23,21},
        {95, 114, 132, 95, 114, 133, 98, 116, 135, 95, 114, 132, 0, 0, 0, 0},
        {38,  76, 113, 39,  58,  76, 95, 114, 132,  1,  21,   0, 0, 0, 0, 0},
        {38,  76, 113, 39,  58,  76, 95, 114, 132,  1,  21,   0, 0, 0, 0, 0},
        {38,  76, 113, 38,  57,  76, 95, 113, 132,  1,  20,   0, 0, 0, 0, 0}
    };

	// hardcoded chamber ph coverage, in prim_conv
    int ph_cover_hard[5][16] =
	{
      {40,40,40,40,40,40,30,30,30,40,40,40,40,40,30,40},
      {40,40,40,40,40,40,30,30,30,40,40,40, 0, 0, 0, 0},
      {80,80,80,40,40,40,40,40,40,80,40, 0, 0, 0, 0, 0},
      {80,80,80,40,40,40,40,40,40,80,40, 0, 0, 0, 0, 0},
      {80,80,80,40,40,40,40,40,40,80,40, 0, 0, 0, 0, 0}
	};

    int th_init_hard[5][16] =
	{
      {1,1,1,42,42,42,94,94,94,1,1, 1,1,42,94, 1},
      {1,1,1,42,42,42,94,94,94,1,1, 1,0, 0, 0, 0},
      {1,1,1,48,48,48,48,48,48,1,48,0,0, 0, 0, 0},
      {1,1,1,40,40,40,40,40,40,1,40,0,0, 0, 0, 0},
      {2,2,2,34,34,34,34,34,34,2,34,0,0, 0, 0, 0}
	};

    for (int sect = 0; sect < 6; sect++) // sector loop
    {
        std::ofstream cham_params, ph_init_fs, th_init_fs, ph_disp_fs, th_disp_fs, ph_init_full_fs;
        ostringstream cham_param_fn, ph_init_fn, th_init_fn, ph_disp_fn, th_disp_fn, ph_init_full_fn;
        cham_param_fn << "cham_param_endcap_"<< endcap <<"_sect_" << (sect+1) << ".h";
        cham_params.open(cham_param_fn.str().c_str());

        ph_init_fn << "ph_init_endcap_" << endcap << "_sect_" << (sect+1) << ".lut";
        ph_init_fs.open(ph_init_fn.str().c_str());

        th_init_fn << "th_init_endcap_" << endcap << "_sect_" << (sect+1) << ".lut";
        th_init_fs.open(th_init_fn.str().c_str());

        ph_disp_fn << "ph_disp_endcap_" << endcap << "_sect_" << (sect+1) << ".lut";
        ph_disp_fs.open(ph_disp_fn.str().c_str());

        th_disp_fn << "th_disp_endcap_" << endcap << "_sect_" << (sect+1) << ".lut";
        th_disp_fs.open(th_disp_fn.str().c_str());

        cham_params << "cham_params chp[5][9] = \n{\n";
        for (int s = 0; s < 5; s++) // station loop
        {

            ph_init_full_fn.str("");
            ph_init_full_fn << "ph_init_full_endcap_" << endcap << "_sect_" << (sect+1) << "_st_" << s << ".lut";
            ph_init_full_fs.open(ph_init_full_fn.str().c_str());
            cham_params << "    {\n";
            for (int cn = 0; cn < ((s == 0) ? 16 : (s == 1) ? 12 : 11); cn++) // chamber loop (includes neighbor sector)
            {
                int c, rsect; // real chamber and sector
                // find real sector and chamber for neighbor sector's chambers
                if (s == 0)
                {
                    // ME1a
                    switch (cn)
                    {
                        case 12: c = 2;  rsect = sect-1; if (rsect == -1) rsect=5; break;
                        case 13: c = 5;  rsect = sect-1; if (rsect == -1) rsect=5; break;
                        case 14: c = 8;  rsect = sect-1; if (rsect == -1) rsect=5; break;
                        case 15: c = 11; rsect = sect-1; if (rsect == -1) rsect=5; break;
                        default: c = cn; rsect = sect;
                    }
                }
                else
                if (s == 1)
                {
                    // ME1b
                    c = cn; rsect = sect; // no neighbor here
                }
                else
                {
                    switch (cn)
                    {
                        case 9:  c = 2; rsect = sect-1; if (rsect == -1) rsect=5; break;
                        case 10: c = 8; rsect = sect-1; if (rsect == -1) rsect=5; break;
                        default: c = cn; rsect = sect;
                    }

                }

                // calculate strip phi pitch factor relative to ME234/2
                double factor = strip_phi_pitch[rsect][s==0? 1:s][c+1]/strip_phi_pitch[rsect][2][4];
                if (factor > 1.9 && factor < 2.1) factor = 2.;
                else if (factor > 1.1 || factor < 0.99) factor *= 1024.;

                int ch_type = c / 3;
                if (s > 1 && ch_type > 1) ch_type = 1; // st 234 have only 2 chamber types
                if (s < 2 && ch_type == 3) ch_type = 0; // ME1/1 is type 0

                cham_params << "        {"
                            << setw(4) << ((int)factor) << ", "
                            << setw(2) << num_of_strips    [rsect][s==0? 1:s][c+1] << ", "
                            << setw(3) << num_of_wiregroups[rsect][s==0? 1:s][c+1] << ", "
                            << setw(3) << ph_init[sect][s][cn] << ", "
                            << setw(3) << th_init[sect][s][cn] << ", "
                            << setw(2) << ph_cover[sect][s][cn] << ", "
                            << setw(2) << th_cover[sect][s][cn] << ", "
                            << setw(1) << (int)(s < 2 && c < 3) << ", "
                            << setw(1) << (int)ph_reverse[sect][s][cn];
                if (c == 8) cham_params << "}\n";
                else        cham_params << "},\n";

                ph_init_fs << hex << ph_init[sect][s][cn] << endl;
                th_init_fs << hex << th_init[sect][s][cn] << endl;

                ph_disp_fs << hex
                           << ((ph_reverse[sect][s][cn]) ? (ph_init[sect][s][cn]/2 - ph_cover_hard[s][cn] - 2*ph_init_hard[s][cn]) :
                                                          (ph_init[sect][s][cn]/2 - 2*ph_init_hard[s][cn])) << endl;

                th_disp_fs << hex << (th_init[sect][s][cn] - th_init_hard[s][cn]) << endl;
                ph_init_full_fs << hex << ph_init_full[sect][s][cn] << endl;

            }
            if (s == 4) cham_params << "    }\n";
            else        cham_params << "    },\n";
            ph_init_full_fs.close();
        }
        cham_params << "};\n";
        cham_params.close();

        ph_init_fs.close();
        th_init_fs.close();
        ph_disp_fs.close();
        th_disp_fs.close();

    }



}



//returns a real number (approx. 0 - 63 degrees) that represents the angular coordinate within each sector
double slhc_geometry::getLocalSectorPhiValue(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId, edm::EventSetup const& es) const 
{

  double globalPhi = getGlobalPhiValue(endcap,  station,  sector,  subsector,  wireGroup,  strip,  cscId, es);

	// sector boundary should not depend on station, cscid, etc. For now, take station 2 csc 1 strip 0 as boundary, -2 deg (Darin, 2009-09-18)
  double sector_start = getGlobalPhiValue(endcap,  2,  sector,  1,  0,  endcap == 1 ? 0:79,  1, es) - 2;
	double res = globalPhi - sector_start;
	// strange effect - global phi is sometimes returned negative, -360 deg from the point that's needed
	if (res < -200.) res += 360.;
	return res;
}

//returns a real number (approx. 0 - 63 degrees) that represents the angular coordinate within each sector
// takes half-strip number instead of strip
double slhc_geometry::get_sector_phi_hs 
(
     const unsigned& endcap, 
     const unsigned& station, 
     const unsigned& sector, 
     const unsigned& subsector, 
     const unsigned& wireGroup, 
     const unsigned& halfstrip, 
     const unsigned& cscId,
        const bool& nb,
     edm::EventSetup const& es
) const 
{
  unsigned ring = 0, ichamber = 0;

  get_ring_chamber(station, sector, subsector, cscId, ring, ichamber);

  CSCDetId id = CSCDetId(endcap, station, ring, ichamber, CSCConstants::KEY_CLCT_LAYER);
  edm::ESHandle<CSCGeometry> cscGeom;
  es.get<MuonGeometryRecord>().get(cscGeom);

  const CSCChamber* chamber = cscGeom->chamber(id);
  CSCLayerGeometry* layerGeom = const_cast<CSCLayerGeometry*>(chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());

  int strip = halfstrip / 2;
  int oddhs = halfstrip % 2;
  
  double globalPhi0 = getGlobalPhiValue(endcap,  station,  sector,  subsector,  wireGroup,  strip,  cscId, es);

  double pitch = layerGeom->stripPhiPitch() * 180. / M_PI;
  int ph_reverse = (endcap == 1 && station >= 3) ? 1 : 
    (endcap == 2 && station <  3) ? 1 : 0;

  if (ph_reverse == 1) pitch = -pitch;

  if (oddhs == 1) globalPhi0 += pitch/4; // take half strip into account
  else            globalPhi0 -= pitch/4;

  unsigned sector_n; // neighbor sector
  if (nb) sector_n = sector; else // if nb flag is true, the user wants coordinates from neighbor sector, do not apply correction
  {
    if (sector > 1) sector_n = sector - 1; else sector_n = 6; // find neighbor sector
  }

  // sector boundary should not depend on station, cscid, etc. For now, take station 2 csc 1 strip 0 as boundary, -2 deg (Darin, 2009-09-18)
  // correction for sector overlap: take sector boundary at previous sector, station 2 csc 3 strip 0 - 2 deg (2016-03-07)
  double sector_start = getGlobalPhiValue(endcap,  2,  sector_n,  1,  0,  endcap == 1 ? 0:79,  3, es) - 2;
  double res = globalPhi0 - sector_start;
  // strange effect - global phi is sometimes returned negative, -360 deg from the point that's needed
  if (res < -200.) res += 360.;
  return res;
}



//returns theta as a double
double slhc_geometry::getTheta
(
    const unsigned& endcap, 
    const unsigned& station, 
    const unsigned& sector, 
    const unsigned& subsector, 
    const unsigned& wireGroup, 
    const unsigned& strip, 
    const unsigned& cscId,
    edm::EventSetup const& es
) const 
{
	
  unsigned ring = 0;
  unsigned ichamber = 0;

  get_ring_chamber(station, sector, subsector, cscId, ring, ichamber);

  CSCDetId id = CSCDetId(endcap, station, ring, ichamber, CSCConstants::KEY_CLCT_LAYER);
  edm::ESHandle<CSCGeometry> cscGeom;
  es.get<MuonGeometryRecord>().get(cscGeom);

  const CSCChamber* chamber = cscGeom->chamber(id);
  CSCLayerGeometry* layerGeom = const_cast<CSCLayerGeometry*>(chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());

	
//    LocalPoint lPoint = layerGeom->stripWireGroupIntersection(strip+1, wireGroup+1);
    LocalPoint lPoint = layerGeom->intersectionOfStripAndWire(strip+1, layerGeom->middleWireOfGroup(wireGroup+1));
    GlobalPoint gPoint = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);
	
	
	if (endcap == 1)
		return gPoint.theta()*180./M_PI;
	else
		return 180. - gPoint.theta()*180./M_PI; //to get a theta value in the range 0 - 180. degrees for endcap 2
	
}

double slhc_geometry::getTheta_limited
(
    const unsigned& endcap,
    const unsigned& station,
    const unsigned& sector,
    const unsigned& subsector,
    const unsigned& wireGroup,
    const unsigned& strip,
    const unsigned& cscId,
    edm::EventSetup const& es
) const
{

  unsigned ring = 0;
  unsigned ichamber = 0;

  get_ring_chamber(station, sector, subsector, cscId, ring, ichamber);

  CSCDetId id = CSCDetId(endcap, station, ring, ichamber, CSCConstants::KEY_CLCT_LAYER);
  edm::ESHandle<CSCGeometry> cscGeom;
  es.get<MuonGeometryRecord>().get(cscGeom);

  const CSCChamber* chamber = cscGeom->chamber(id);
  CSCLayerGeometry* layerGeom = const_cast<CSCLayerGeometry*>(chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());


//    LocalPoint lPoint = layerGeom->stripWireGroupIntersection(strip+1, wireGroup+1);
    LocalPoint lPoint = layerGeom->intersectionOfStripAndWire(strip+1, layerGeom->middleWireOfGroup(wireGroup+1));
    GlobalPoint gPoint = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);


    if (endcap == 1)
        return gPoint.theta()*180./M_PI;
    else
        return 180. - gPoint.theta()*180./M_PI; //to get a theta value in the range 0 - 180. degrees for endcap 2

}


//returns theta as a double
double slhc_geometry::getTheta_wire(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wire, const unsigned& strip, const unsigned& cscId) const {
	
	bool isTMB07 = true;

	CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
	CSCChamber* thechamber = NULL;
	CSCLayerGeometry* layerGeom = NULL;

	thechamber = thegeom->chamber(endcap,station,sector,subsector,cscId);
	if(thechamber) {
	
		if(isTMB07) {
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());
	    	}
	  	else {
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER_PRE_TMB07)->geometry());
	    	}
	}
		
	
	LocalPoint lPoint = layerGeom->stripWireIntersection(strip+1, wire+1);
	GlobalPoint gPoint = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);
	
	
	if (endcap == 1)
		return gPoint.theta()*180./M_PI;
	else
		return 180. - gPoint.theta()*180./M_PI; //to get a theta value in the range 0 - 180. degrees for endcap 2
	
}

// convert trigger-style variables into geometry-style
void  slhc_geometry::get_ring_chamber
(
    const unsigned& station, 
    const unsigned& sector, 
    const unsigned& subsector, 
    const unsigned& cscId, 
    unsigned& ring, 
    unsigned& ichamber
    ) const
{
  switch (station)
    {
    case 1:
      ring = (cscId - 1)/3 + 1;

      ichamber = (cscId - 1)%3 + (subsector-1)*3 + (sector-1)*6 + 2;
      ichamber = ichamber % 36 + 1;

      break;
    case 2:
    case 3:
    case 4:
      if (cscId > 3) 
	{
	  ring = 2;
	  ichamber = (cscId - 4) + (sector-1)*6 + 2;
	  ichamber = ichamber % 36 + 1 ;
	}
      else
	{ 
	  ring = 1;
	  ichamber = (cscId - 1) + (sector-1)*3 + 1;
	  ichamber = ichamber % 18 + 1;
	}
      
      break;

    }
}

//returns phi in global coordinates (in degrees) as a double
// uses core geometry routines as opposed to trigger geometry function just below this one
double slhc_geometry::getGlobalPhiValue
(
    const unsigned& endcap, 
    const unsigned& station, 
    const unsigned& sector, 
    const unsigned& subsector, 
    const unsigned& wireGroup, 
    const unsigned& strip, 
    const unsigned& cscId, 
    edm::EventSetup const& es
) const {
	
  
  unsigned ring = 0;
  unsigned ichamber = 0; 


  get_ring_chamber(station, sector, subsector, cscId, ring, ichamber);

  CSCDetId id = CSCDetId(endcap, station, ring, ichamber, CSCConstants::KEY_CLCT_LAYER);
  edm::ESHandle<CSCGeometry> cscGeom;
  es.get<MuonGeometryRecord>().get(cscGeom);

  const CSCChamber* chamber = cscGeom->chamber(id);
  CSCLayerGeometry* layerGeom = const_cast<CSCLayerGeometry*>(chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());

  LocalPoint lPoint = layerGeom->stripWireGroupIntersection(strip+1, wireGroup+1); // strip and wg in geometry routines start from 1
  GlobalPoint gPoint = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);
	

  double result = gPoint.phi();

  // phi is returned from -180 to +180, make it always positive
  if (result < 0.) result += 2 * M_PI;


  return result*180./M_PI;

}

/*
//returns phi in global coordinates (in degrees) as a double
double slhc_geometry::getGlobalPhiValue(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId) const {
	
  	bool isTMB07 = true;

	CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
	CSCChamber* thechamber = NULL;
	CSCLayerGeometry* layerGeom = NULL;

	thechamber = thegeom->chamber(endcap,station,sector,subsector,cscId);
	if(thechamber) {
		if(isTMB07) {
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());
	    	}
	  	else {
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER_PRE_TMB07)->geometry());
	    	}
	}
	
	LocalPoint lPoint = layerGeom->stripWireGroupIntersection(strip+1, wireGroup+1); // strip and wg in geometry routines start from 1
	GlobalPoint gPoint = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);
	

	double result = gPoint.phi();


	return result*180./M_PI;

}
*/
/*double slhc_geometry::getLocalPhiValue(const unsigned& endcap, const unsigned& station, const unsigned& sector, const unsigned& subsector, const unsigned& wireGroup, const unsigned& strip, const unsigned& cscId) const {
	
	CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
	CSCChamber* thechamber = NULL;
// 	CSCLayer* thelayer = NULL;
	CSCLayerGeometry* layerGeom = NULL;

	thechamber = thegeom->chamber(endcap,station,sector,subsector,cscId);
	if(thechamber) {
		if(isTMB07) {
// 			std::cout << "isTMB07" << std::endl;
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry());
	    	}
	  	else {
			layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_CLCT_LAYER_PRE_TMB07)->geometry());
// 			std::cout << "! isTMB07" << std::endl;
	    	}
	}
// 	else std::cout << "!thechamber" << std::endl;
		
	
	LocalPoint lPoint = layerGeom->stripWireGroupIntersection(strip, wireGroup);
	GlobalPoint gPoint = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);
	
// 	if (endcap == 1)
	double result = lPoint.phi();
// 		std::cout << result << "  " << result*180./M_PI << std::endl;
// 		return gPoint.phi()*180./M_PI;
	return result*180./M_PI;
// 	else
// 		return 180. - gPoint.theta()*180./M_PI;

}*/

