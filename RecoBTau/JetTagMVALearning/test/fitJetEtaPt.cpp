#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>

#include <TFile.h>
#include <TF1.h>
#include <TH1.h>
#include <TH2.h>

double compute(double params[7][6], double x, double y)
{
	double facs[7];
	for(int i = 0; i < 7; i++) {
		double *v = params[i];
		facs[i] = v[0] + y * (v[1] + y * (v[2] + y * (v[3] + y * (v[4] + y * v[5]))));
	}

	double xs[5];
	xs[0] = x * x;
	xs[1] = xs[0] * xs[0];
	xs[2] = xs[1] * xs[0];
	xs[3] = xs[1] * xs[1];
	xs[4] = xs[2] * xs[1];
	xs[5] = xs[2] * xs[2];

	return facs[0] +
	       facs[1] * (2 * xs[0] - 1) +
	       facs[2] * (8 * xs[1] - 8 * xs[0] + 1) +
	       facs[3] * (32 * xs[2] - 48 * xs[1] + 18 * xs[0] - 1) +
	       facs[4] * (128 * xs[3] - 256 * xs[2] + 160 * xs[1] - 32 * xs[0] + 1) +
	       facs[5] * (512 * xs[4] - 1280 * xs[3] + 1120 * xs[2] - 400 * xs[1] + 50 * xs[0] - 1) +
	       facs[6] * (2048 * xs[5] - 6144 * xs[4] + 6912 * xs[3] - 3584 * xs[2] + 840 * xs[1] - 72 * xs[0] + 1);
}

int main(int argc, char **argv)
{
	assert(argc == 2 || argc == 3);

	std::auto_ptr<TFile> inFile(new TFile(argv[1]));
	assert(inFile.get());
	TH2D *th2 = dynamic_cast<TH2D*>(inFile->Get("jets"));
	assert(th2);
	th2->Sumw2();

	if (argc == 3) {
		std::auto_ptr<TFile> inFile2(new TFile(argv[2]));
		assert(inFile2.get());
		TH2D *th2b = dynamic_cast<TH2D*>(inFile2->Get("jets"));
		assert(th2b);
		th2b->Sumw2();
		th2->Divide(th2b);
	}

	TH1D *th1 = new TH1D("slice", "slice", 50, -1.0, +1.0);
	th1->SetDirectory(0);

	TF1 *cheb = new TF1("ChebS8", "[0] +"
	                              "[1] * (2 * (x^2) - 1) + "
	                              "[2] * (8 * (x^4) - 8 * (x^2) + 1) + "
	                              "[3] * (32 * (x^6) - 48 * (x^4) + 18 * (x^2) - 1) + "
	                              "[4] * (128 * (x^8) - 256 * (x^6) + 160 * (x^4) - 32 * (x^2) + 1) +"
	                              "[5] * (512 * (x^10) - 1280 * (x^8) + 1120 * (x^6) - 400 * (x^4) + 50 * (x^2) - 1) +"
	                              "[6] * (2048 * (x^12) - 6144 * (x^10) + 6912 * (x^8) - 3584 * (x^6) + 840 * (x^4) - 72 * (x^2) + 1)",
	                    -1.0, +1.0);
	cheb->SetParLimits(0, 0.0, 100000.0);

	static const int max = 38;

	TH1D *coeffs[7];
	for(int i = 0; i < 7; i++) {
		coeffs[i] = new TH1D(Form("coeff%d", i), "coeffs", max, 0.5, max + 0.5);
		coeffs[i]->SetDirectory(0);
	}

	for(int y = 1; y <= max; y++) {
		for(int x = 1; x <= 50; x++)
			th1->SetBinContent(x, th2->GetBinContent(x, y));
		th1->Fit(cheb, "QRNB");
		for(int i = 0; i < 7; i++) {
			coeffs[i]->SetBinContent(y, cheb->GetParameter(i));
			coeffs[i]->SetBinError(y, cheb->GetParError(i));
		}
	}

	double params[7][6];

	for(int i = 0; i < 7; i++) {
		coeffs[i]->Fit("expo", "Q0");
		TF1 *f1 = coeffs[i]->GetFunction("expo");
		double expo, arg;
		if (f1->GetParameter(0) >= -100000.0) {
			expo = std::exp(f1->GetParameter(0));
			arg = f1->GetParameter(1);
		} else {
			coeffs[i]->Scale(-1.0);
			coeffs[i]->Fit("expo", "Q0");
			f1 = coeffs[i]->GetFunction("expo");
			expo = -std::exp(f1->GetParameter(0));
			arg = f1->GetParameter(1);
			coeffs[i]->Scale(-1.0);
		}
		coeffs[i]->Fit("pol5", "0B");
		TF1 *pol = coeffs[i]->GetFunction("pol5");
		for(int j = 0; j < 6; j++)
			params[i][j] = pol->GetParameter(j);
	}

	double chi2 = 0.0;
	int ndf = 0;
	TH2D *th2c = new TH2D(*th2);
	th2c->SetName("fit");
	for(int y = 1; y <= 40; y++) {
		int ry = y > 36 ? 36 : y;
		for(int x = 1; x <= 50; x++) {
			double val = compute(params, (x - 0.5) / 25.0 - 1.0, ry);
			if (val < 0)
				val = 0;
			double error = th2->GetBinError(x, y);
			if (error > 0 && val > 0) {
				double chi = th2->GetBinContent(x, y) - val;
				chi2 += chi * chi / val;
				ndf++;
			}
			th2c->SetBinContent(x, y, val); // - th2->GetBinContent(x, y));
		}
	}
	std::cout << "chi2/ndf(" << ndf << "): " << (chi2 / ndf) << std::endl;

	TFile g("out.root", "RECREATE");
	th2->SetDirectory(&g);
	th2->Write();
	th2c->SetDirectory(&g);
	th2c->Write();
	g.Close();

	std::ofstream of("out.txt");
	of << std::setprecision(16);
	for(int i = 0; i < 7; i++)
		for(int j = 0; j < 6; j++)
			of << params[i][j] << (j < 5 ? "\t" : "\n");
	of.close();

	return 0;
}
