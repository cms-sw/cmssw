import root;
import pad_layout;

string topDir = "../";

string n_events = "1E4";
string simulations[] = {
	"simulations/"+n_events+"/vtx_y,ang,xi.root",
	"simulations/"+n_events+"/vtx,ang,xi.root",
	"simulations/"+n_events+"/vtx,ang,xi,det.root",
	"simulations/"+n_events+"/vtx,ang,xi,det,bd.root",
};

string sectors[] = {
	"45",
	"56",
};

//----------------------------------------------------------------------------------------------------

void PlotAll(string f, string sector, string hist, real x_scale=1, real y_scale=1)
{
	RootObject obj = RootGetObject(f, "sector " + sector + "/" + hist + "_" + sector);
	draw(scale(x_scale, y_scale), obj, "def");
}

//----------------------------------------------------------------------------------------------------

for (string simulation : simulations)
{
	for (string sector : sectors)
	{
		NewRow();
	
		NewPad(false);
		label(replace("\vbox{\SetFontSizesXX\hbox{" + simulation + "}\hbox{sector " + sector + "}}", "_", "\_"));
	
		string f = topDir + simulation;
	
		//--------------------
	
		NewPad("$\De\xi$", "$\De x^{*}\ung{\mu m}$");
		scale(Linear, Linear, Log);
		PlotAll(f, sector, "h2_de_vtx_x_vs_de_xi", 1., 1e6);
		limits((-0.005, -40), (+0.005, +40), Crop);
		AttachLegend(BuildLegend(lineLength=5mm));
	
		//--------------------
	
		NewPad("$\De\xi$", "$\De y^{*}\ung{\mu m}$");
		scale(Linear, Linear, Log);
		PlotAll(f, sector, "h2_de_vtx_y_vs_de_xi", 1., 1e6);
		limits((-0.005, -250), (+0.005, +250), Crop);
		AttachLegend(BuildLegend(lineLength=5mm));
	
		//--------------------
	
		NewPad("$\De\xi$", "$\De \th_x^{*}\ung{\mu rad}$");
		scale(Linear, Linear, Log);
		PlotAll(f, sector, "h2_de_th_x_vs_de_xi", 1., 1e6);
		limits((-0.005, -100), (+0.005, +100), Crop);
		AttachLegend(BuildLegend(lineLength=5mm));
	
		//--------------------
	
		NewPad("$\De\xi$", "$\De \th_y^{*}\ung{\mu rad}$");
		scale(Linear, Linear, Log);
		PlotAll(f, sector, "h2_de_th_y_vs_de_xi", 1., 1e6);
		limits((-0.005, -100), (+0.005, +100), Crop);
		AttachLegend(BuildLegend(lineLength=5mm));
	
		//--------------------
	
		NewPad("$\De\th_y\ung{\mu rad}$", "$\De y^{*}\ung{\mu m}$");
		scale(Linear, Linear, Log);
		PlotAll(f, sector, "h2_de_vtx_y_vs_de_th_y", 1e6, 1e6);
		limits((-100, -250), (+100, +250), Crop);
		AttachLegend(BuildLegend(lineLength=5mm));
	}
}
