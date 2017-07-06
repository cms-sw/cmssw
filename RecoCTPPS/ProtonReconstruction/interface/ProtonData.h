/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef _proton_data_h_
#define _proton_data_h_

struct ProtonData
{
	bool valid = false;

	double vtx_x = 0., vtx_y = 0.;	// m

	double th_x = 0., th_y = 0.;	// rad

	double xi = 0.;					// 1, positive when energy loss
	double xi_unc = 0.;

	void Print() const
	{
		printf("valid=%i, vtx_x=%.3f mm, vtx_y=%.3f mm, th_x=%.1f urad, th_y=%.1f urad, xi=%.3f\n",
			valid, vtx_x*1E3, vtx_y*1E3, th_x*1E6, th_y*1E6, xi);
	}
};

#endif
