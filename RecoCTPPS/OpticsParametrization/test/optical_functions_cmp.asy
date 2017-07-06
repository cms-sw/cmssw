import root;
import pad_layout;

string old_dir = "/afs/cern.ch/user/j/jkaspar/software/ctpps_proton_reconstruction/";

string rp_tags[], rp_labels[], rp_old_files[], rp_new_files[], rp_new_objs[];
rp_tags.push("3"); rp_labels.push("45-210-fr"); rp_old_files.push("get_optical_functions_300E-6.root"); rp_new_files.push("optical_functions_45.root"); rp_new_objs.push("ip5_to_station_150_h_2_lhcb2");
rp_tags.push("2"); rp_labels.push("45-210-nr"); rp_old_files.push("get_optical_functions_300E-6.root"); rp_new_files.push("optical_functions_45.root"); rp_new_objs.push("ip5_to_station_150_h_1_lhcb2");
rp_tags.push("102"); rp_labels.push("56-210-nr"); rp_old_files.push("get_optical_functions_200E-6.root"); rp_new_files.push("optical_functions_56.root"); rp_new_objs.push("ip5_to_station_150_h_1_lhcb1");
rp_tags.push("103"); rp_labels.push("56-210-fr"); rp_old_files.push("get_optical_functions_200E-6.root"); rp_new_files.push("optical_functions_56.root"); rp_new_objs.push("ip5_to_station_150_h_2_lhcb1");

//----------------------------------------------------------------------------------------------------

void PlotAll(int rpi, string obj)
{
	draw(scale(1., +1e3), RootGetObject(old_dir + rp_old_files[rpi], "RP"+rp_tags[rpi]+"/" + obj), blue);
	draw(scale(1., +1e3), RootGetObject(rp_new_files[rpi], rp_new_objs[rpi]+"/" + obj), red+dashed);
}

//----------------------------------------------------------------------------------------------------

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	NewPad("$\xi$", "$x_0\ung{mm}$");
	PlotAll(rpi, "g_x0_vs_xi");

	//--------------------

	NewPad("$\xi$", "$y_0\ung{mm}$");
	PlotAll(rpi, "g_y0_vs_xi");

	//--------------------

	NewPad("$\xi$", "$v_x$");
	PlotAll(rpi, "g_v_x_vs_xi");

	//--------------------

	NewPad("$\xi$", "$v_y$");
	PlotAll(rpi, "g_v_y_vs_xi");

	//--------------------

	NewPad("$\xi$", "$L_x\ung{mm}$");
	PlotAll(rpi, "g_L_x_vs_xi");

	//--------------------

	NewPad("$\xi$", "$L_y\ung{mm}$");
	PlotAll(rpi, "g_L_y_vs_xi");

	//--------------------

	NewPad(false);

	AddToLegend("before CMSSW integration", blue);
	AddToLegend("after CMSSW integration", red+dashed);

	AttachLegend();
}
