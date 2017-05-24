import root;
import pad_layout;

string rp_tags[], rp_labels[];
//rp_tags.push("3"); rp_labels.push("45-210-fr");
//rp_tags.push("2"); rp_labels.push("45-210-nr");
rp_tags.push("102"); rp_labels.push("56-210-nr");
//rp_tags.push("103"); rp_labels.push("56-210-fr");


string dir_simu = "../simulations/";
string dir_data = "/afs/cern.ch/work/j/jkaspar/analyses/ctpps/alignment/";

string files[], f_types[], f_labels[];
files.push(dir_simu + "1E5/vtx,ang,xi,det,bd.root"); f_types.push("simulation"); f_labels.push("vtx,ang,xi,det,bd");
files.push(dir_data + "period1_alignment/10077/reconstruction_test.root"); f_types.push("data -- alignment fill"); f_labels.push("run 10077");
//files.push(dir_data + "period1_physics_margin/fill_4947/reconstruction_test.root"); f_types.push("data"); f_labels.push("fill 4947");
//files.push(dir_data + "period1_physics/fill_5261/reconstruction_test.root"); f_types.push("data"); f_labels.push("fill 5261");

//----------------------------------------------------------------------------------------------------

NewPad(false);
for (int fi : files.keys)
{
	NewPad(false);
	label("\vbox{\SetFontSizesXX\hbox{"+f_types[fi]+"}\hbox{"+f_labels[fi]+"}}");
}

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	for (int fi : files.keys)
	{
		string x_axis_label;
		string objPath;
		transform t;

		if (f_types[fi] == "simulation")
		{
			x_axis_label = "$x_{\rm LHC}\ung{mm}$";
			objPath = "h2_y_vs_x_RP"+rp_tags[rpi];
			t = scale(1e3, 1e3);
		} else {
			x_axis_label = "$x_{\rm RP,aligned}\ung{mm}$";
			objPath = "method x/with cuts/h2_y_vs_x_"+rp_tags[rpi];
			t = scale(1, 1);
		}

		NewPad(x_axis_label, "$y\ung{mm}$");
		scale(Linear, Linear, Log);
	
		draw(t, RootGetObject(files[fi], objPath));

		limits((0, -15), (20, +15), Crop);
	}
}

GShipout(vSkip=1mm);
