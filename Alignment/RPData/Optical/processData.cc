#include "TH1D.h"
#include "TFile.h"

#include <map>
#include <vector>
#include <string>

using namespace std;

struct RPData {
	string placement;
	vector<unsigned int> hybrids;
	bool inverted;
};

// map: RP Id -> list of hybrids
map<unsigned int, RPData> rpList;


//----------------------------------------------------------------------------------------------------

struct align
{
	//double lx, ly, rx, ry;
	float x1, y1, x2, y2;

	float ControlDistance() const
		{ return sqrt((x2 - x1)*(x2 - x1) + (y2-y1)*(y2-y1)); }
};

// map: hybrid id --> alignments
map<unsigned int, align> alignments;

//----------------------------------------------------------------------------------------------------

void processData()
{
	// load RP hybrid mapping
	FILE *file = fopen("rp_mapping", "r");
	while (true) {
		unsigned int rp_id = 0;
		char placement[30];
		unsigned int first_hyb = 0;
		RPData d;
		
		fscanf(file, "%u%s", &rp_id, placement);
		for (unsigned int i = 0; i < 10; i++) {
			unsigned int hyb_id = 0;
			fscanf(file, "%u", &hyb_id);
			d.hybrids.push_back(hyb_id);
		}
		fscanf(file, "%u", &first_hyb);

		if (feof(file))
			break;

		d.placement = placement;
		d.inverted = (first_hyb != d.hybrids[0]);

		rpList[rp_id] = d;
	}
	fclose(file);

	TFile *statF = new TFile("statistics.root", "recreate");
	TH1D *cdH = new TH1D("cdH", "control distance histogram", 100, 49.12, 49.15);

	// load data
	file = fopen("data", "r");
	while (true) {
		unsigned int hid = 0;
		align a;
		fscanf(file, "%u%f%f%f%f", &hid, &a.x1, &a.y1, &a.x2, &a.y2);

		if (feof(file))
			break;

		//printf("%u\t%f\t%f\t%f\t%f\n", hid, x1, y1, x2, y2);
		cdH->Fill(a.ControlDistance());

		alignments[hid] = a;
	}
	fclose(file);

	cdH->Write();
	delete statF;

	// theoretical values
	double x_th = (75.068 + 25.932) / 2;
	double y_th = (31.631 + 31.631) / 2;


	for (map<unsigned int, RPData>::iterator it = rpList.begin(); it != rpList.end(); ++it) {
		char buf[20];
		sprintf(buf, "DP%u.xml", it->first);
		file = fopen(buf, "w");
		fprintf(file, "<!-- DP #%u at %s -->\n", it->first, it->second.placement.c_str());
		fprintf(file, "<!--Shifts in um, rotations in mrad. -->\n");
		fprintf(file, "<xml> DocumentType=\"AlignmentDescription\"\n");

		printf("------------------ DP %u -----------------\n", it->first);

		for (unsigned int i = 0; i < 10; i++) {
			unsigned int idx = (it->second.inverted) ? 9 - i : i;
			map<unsigned int, align>::iterator hybIt = alignments.find(it->second.hybrids[idx]);
			unsigned int detId = 1200 + i;
			if (hybIt != alignments.end()) {
			
				align al = hybIt->second;
				
				//printf("%f\t%f\t%f\t%f\n", al.x1, al.y1, al.x2, al.y2);

				// back-to-back correction
				if (idx % 2 == 1) {
					al.y1 = - al.y1;
					al.y2 = - al.y2;
					swap(al.x2, al.x1);
				}
	
				printf("%u (%u)\t%3u\t%f\t%f\t%f\t%f\n", i, idx, hybIt->first, al.x1, al.y1, al.x2, al.y2);
	
				// TODO: check the formulae
				double rot_z = (fabs(al.y1) - fabs(al.y2)) / (al.x1 - al.x2);
				double sh_x = (al.x1 + al.x2) / 2. - x_th;
				double sh_y = fabs(al.y1 + al.y2) / 2. - y_th;

				fprintf(file, "\t<det id=\"%u\" sh_x=\"%+.1f\" sh_y=\"%+.1f\" rot_z=\"%+.2f\"/>\n", detId, sh_x*1E3, sh_y*1E3, rot_z*1E3);
			} else {
				fprintf(file, "\t<det id=\"%u\"/>\n", detId);
			}
		}

		fprintf(file, "</xml>\n");
		fclose(file);
	}

#if 0
	for (map<unsigned int, vector<unsigned int> >::iterator rpit = rp_hybrids.begin(); rpit != rp_hybrids.end(); ++rpit) {
		printf("RP %u\n", rpit->first);

		caaahar buf[20];
		sprintf(buf, "align/RP%u_optical.align", rpit->first);
		FILE *file = fopen(buf, "w");

		unsigned int idx = 0;
		for (vector<unsigned int>::iterator hybit = rpit->second.begin(); hybit != rpit->second.end(); ++hybit, ++idx) {
			printf("\t%u\t", *hybit);

			map<unsigned int, align>::iterator alit = alignments.find(*hybit);
			if (alit == alignments.end()) {
				printf("alignments not found\n");
				continue;
			}

			align al = alit->second;

			if (idx % 2 == 1) {
				al.ly = - al.ly;
				al.ry = - al.ry;
				double temp = al.rx;
				al.rx = al.lx;
				al.lx = temp;
			}

			printf("%f\t%f\t%f\t%f\n", al.lx, al.ly, al.rx, al.ry);

			// displacement of the detector
			double rot = (fabs(al.ly) - fabs(al.ry)) / (al.lx - al.rx);
			double sh_x = (al.lx + al.rx) / 2. - x_th;
			double sh_y = fabs(al.ly + al.ry) / 2. - y_th;

			printf("\t\t%u\t%.2f\t%.2f\t%.2f\n", idx, sh_x*1E3, sh_y*1E3, rot*1E3);

			fprintf(file, "%u\t%.2f\t%.2f\t%.2f\n", 1200 + idx, sh_x*1E3, sh_y*1E3, rot*1E3);
		}

		fclose(file);
	}
#endif
}
