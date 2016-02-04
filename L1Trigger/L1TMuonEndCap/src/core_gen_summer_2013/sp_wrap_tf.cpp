#include <iomanip>
#include <string.h>
#include "sp_wrap.h"


sp_wrap spw;
extern signal_ stdout_sig;
unsigned quality[5][9][2];
unsigned wiregroup[5][9][2];
unsigned halfstrip[5][9][2];
unsigned clct_pattern[5][9][2];

	unsigned bt_phi [3];
	unsigned bt_theta [3];
	unsigned bt_cpattern [3];
	// ph and th deltas from best stations
	// [best_track_num]; last index: [0] - best pair of stations; [1] - second best pair
	unsigned bt_delta_ph [3][2];
	unsigned bt_delta_th [3][2]; 
	unsigned bt_sign_ph[3][2];
	unsigned bt_sign_th[3][2];
	// ranks [best_track_num]
	unsigned bt_rank [3];
	// segment IDs
	// [best_track_num][station 0-3]
	unsigned bt_vi [3][5]; // valid
	unsigned bt_hi [3][5]; // bx index
	unsigned bt_ci [3][5]; // chamber
	unsigned bt_si [3][5]; // segment


#define seg_ch 2

// primitive counters
int pr_cnt[5][9];

int read_event(ifstream &inf);

int main(int argc, char* argv[])
{

	int ip, iev = 0;
	memset (pr_cnt, 0, sizeof (pr_cnt));

	ifstream inf(argv[1]);

	int cn = 0;

	while (inf.good())
    {
		// clean inputs
		for (int s = 0; s < 5; s++) // station
			for (int c = 0; c < 9; c++) // chamber
				for (int p = 0; p < seg_ch; p++) // primitive
				{
					quality[s][c][p] = 0;
					wiregroup[s][c][p] = 0;
					halfstrip[s][c][p] = 0;
					clct_pattern[s][c][p] = 0;
				}

		if (read_event(inf) == -1) break;

		cn++;

		spw.run
			(
				quality,
				wiregroup,
				halfstrip,
				clct_pattern,

				bt_phi ,
				bt_theta ,
				bt_cpattern,
				bt_delta_ph ,
				bt_delta_th ,
				bt_sign_ph,
				bt_sign_th,
				bt_rank ,
				bt_vi ,
				bt_hi ,
				bt_ci ,
				bt_si 

				);


				printf ("event: %d\n",  iev++);


				for (ip = 0; ip < 3; ip = ip+1)
				{
					if (bt_rank[ip] != 0)
						printf ("track: %d  rank: %x ph_deltas: %d %d  th_deltas: %d %d  phi: %d,  theta: %d cpat: %d\n",  
								ip,  bt_rank[ip],  bt_delta_ph[ip][0],  bt_delta_ph[ip][1],  
								bt_delta_th[ip][0],  bt_delta_th[ip][1],  bt_phi[ip],  bt_theta[ip], bt_cpattern[ip]);
				}
    }
	inf.close();
}

int read_event(ifstream &inf)
{
	char line[1000];
	int v[5];
	int _event = -1, _endcap = -1, _sector = -1, _subsector = -1, _station = -1;
	int _valid = -1, _quality = -1, _pattern = -1, _wiregroup = -1;
	int _cscid = -1, _bend = -1, _halfstrip = -1;

	///Temporary workaround for compiler complain on unused variables. By AK.
	_endcap+=_bend+_valid;
	_endcap=-1;
	/////////////////

	// primitive counters for each chamber

	while (inf.good())
	{
		inf.getline(line, sizeof(line));

		// reset v to illegal values
		memset (v, 0xff, sizeof (v));

		// read values
		int sn = sscanf(line, "%d %d %d %d %d", &v[0], &v[1], &v[2], &v[3], &v[4]);
		switch (sn)
		{
			case 1: 
			{	// end of event
				// clean primitive counters
				memset (pr_cnt, 0, sizeof (pr_cnt));
//				cout << "end of event " << dec << _event << endl;
				return 0;
			}
			case 5: 
			{	// first line of primitive
				_event = v[0];	
				_endcap = v[1];
				_sector = v[2];
				_subsector = v[3];
				_station = v[4];
				if (_station == 1 && _subsector == 1) _station = 0;
				break;
			}
			case 4: 
			{   // second line of primitive
				_valid = v[0];	
				_quality = v[1];
				_pattern = v[2];
				_wiregroup = v[3];
				break;
			}
			case 3: 
			{	// last line of primitive
				_cscid = v[0]-1;	
				_bend = v[1];
				_halfstrip = v[2];

				// copy data to the corresponding input
				
				if (pr_cnt[_station][_cscid] >= seg_ch)
				{
				    cout << "bad segment index. event: " << dec << _event << " line: '" << line << "' index: " << pr_cnt[_station][_cscid] << " station: " << _station << " cscid: " << _cscid << endl;
				}
				else
				{
					if (_station < 0 || _station > 4) cout << "bad station: " << dec << _station << endl;
					else if (_sector != 1) cout << "bad sector: " << dec << _sector << endl;
					else if (_cscid < 0 || _cscid > 8) cout << "bad cscid: " << dec << (_cscid+1) << " station: " << _station << endl;
					else
					{

						quality[_station][_cscid][pr_cnt[_station][_cscid]] = _quality;
						wiregroup[_station][_cscid][pr_cnt[_station][_cscid]] = _wiregroup;
						halfstrip[_station][_cscid][pr_cnt[_station][_cscid]] = _halfstrip;
						clct_pattern[_station][_cscid][pr_cnt[_station][_cscid]] = _pattern;
						pr_cnt[_station][_cscid]++;
					}
				}
				break;
			}
		}
	}

	if (!inf.good()) return -1;

	return 0;
	
}


