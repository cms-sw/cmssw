#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "VoronoiAlgorithm.h"

namespace {

	double hermite_h_normalized(const size_t n, const double x)
	{
		double y;

		switch (n) {
		case 3: y = -3.913998411780905*x + 2.6093322745206033*std::pow(x,3); break;
		case 5: y = 4.931174490213579*x - 6.574899320284771*std::pow(x,3) + 1.3149798640569543*std::pow(x,5); break;
		case 7: y = -5.773117374387059*x + 11.546234748774118*std::pow(x,3) - 4.618493899509647*std::pow(x,5) + 0.43985656185806166*std::pow(x,7); break;
		case 9: y = 6.507479403136423*x - 17.353278408363792*std::pow(x,3) + 10.411967045018276*std::pow(x,5) - 1.9832318180987192*std::pow(x,7) + 0.11017954544992885*std::pow(x,9); break;
		case 11: y = -7.167191940825306*x + 23.89063980275102*std::pow(x,3) - 19.112511842200817*std::pow(x,5) + 5.460717669200234*std::pow(x,7) - 0.6067464076889149*std::pow(x,9) + 0.02206350573414236*std::pow(x,11); break;
		case 13: y = 7.771206704387521*x - 31.084826817550084*std::pow(x,3) + 31.084826817550084*std::pow(x,5) - 11.841838787638126*std::pow(x,7) + 1.9736397979396878*std::pow(x,9) - 0.14353743985015913*std::pow(x,11) + 0.0036804471756451056*std::pow(x,13); break;
		case 15: y = -8.331608118589472*x + 38.88083788675087*std::pow(x,3) - 46.65700546410104*std::pow(x,5) + 22.217621649571925*std::pow(x,7) - 4.9372492554604275*std::pow(x,9) + 0.5386090096865921*std::pow(x,11) - 0.027620974855722673*std::pow(x,13) + 0.00052611380677567*std::pow(x,15); break;
		case 17: y = 8.856659222944476*x - 47.23551585570387*std::pow(x,3) + 66.12972219798543*std::pow(x,5) - 37.7884126845631*std::pow(x,7) + 10.496781301267527*std::pow(x,9) - 1.5268045529116403*std::pow(x,11) + 0.11744650407012618*std::pow(x,13) - 0.004474152536004807*std::pow(x,15) + 0.0000657963608236001*std::pow(x,17); break;
		default: y = 0;
		}

		return y;
	}

}


		VoronoiAlgorithm::VoronoiAlgorithm(
			const UECalibration *ue_,
			const double dr_max,
			const bool exclude_v1_, const int max_vn_, const bool diagonal_vn_,
			const std::pair<double, double> equalization_threshold,
			const bool remove_nonpositive)
			: GenericVoronoiAlgorithm(dr_max, equalization_threshold,
									  remove_nonpositive),
			  ue(ue_),
			  exclude_v1(exclude_v1_), max_vn(max_vn_), diagonal_vn(diagonal_vn_)
		{
		}

		void VoronoiAlgorithm::subtract_momentum(void)
		{
			for (std::vector<particle_t>::iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				int predictor_index = -1;
				int interpolation_index = -1;
				double density = 0;
				double pred_0 = 0;

				for (size_t l = 1; l < _edge_pseudorapidity.size(); l++) {
					if (iterator->momentum.Eta() >=
						_edge_pseudorapidity[l - 1] &&
						iterator->momentum.Eta() <
						_edge_pseudorapidity[l]) {
						predictor_index = l - 1;
					}
				}

				for (size_t j = 0; j < nreduced_particle_flow_id; j++) {
				if (j == 2) {
					// HCAL
					for (size_t l = 1;
						 l < _cms_hcal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.Eta() >=
							_cms_hcal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.Eta() <
							_cms_hcal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}
				else {
					// Tracks or ECAL clusters
					for (size_t l = 1;
						 l < _cms_ecal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.Eta() >=
							_cms_ecal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.Eta() <
							_cms_ecal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}

				if (predictor_index >= 0 && interpolation_index >= 0) {
					// Calculate the aggregated prediction and
					// interpolation for the pseudorapidity segment

					const double azimuth = iterator->momentum.Phi();
					const float (*p)[2][82] =
#ifdef STANDALONE
						ue_predictor_pf[j][predictor_index]
#else // STANDALONE
						ue->ue_predictor_pf[j][predictor_index]
#endif // STANDALONE
						;
					double pred = 0;

					for (size_t l = 0; l < nfourier; l++) {
						const size_t norder = l == 0 ? 9 : 1;

						for (size_t m = 0; m < 2; m++) {
							float u = p[l][m][0];

							for (size_t n = 0; n < 2 * nfourier - 1; n++) {

								if ((!exclude_v1 || l == 1) &&
									(!diagonal_vn || ((l == 0 && n == 0) || (n == 2 * l - 1 || n == 2 * n))) &&
									l <= max_vn) {
									u += p[l][m][9 * n + 1] * _feature[n];
									for (size_t o = 2; o < norder + 1; o++) {
										u += p[l][m][9 * n + o] *
											hermite_h_normalized(
											2 * o - 1, _feature[n]) *
											exp(-_feature[n] * _feature[n]);
									}
								}
							}

							pred += u * (l == 0 ? 1.0 : 2.0) *
								(m == 0 ? cos(l * azimuth) :
								 sin(l * azimuth));
							if (l == 0 && m == 0) {
								pred_0 += u /
									(2.0 * M_PI *
									 (_edge_pseudorapidity[predictor_index + 1] -
									  _edge_pseudorapidity[predictor_index]));
							}
						}
					}

					double interp = 0;

#ifdef STANDALONE
					if (j == 0) {
						interp =
							ue_interpolation_pf0[predictor_index][
								interpolation_index];
					}
					else if (j == 1) {
						interp =
							ue_interpolation_pf1[predictor_index][
								interpolation_index];
					}
					else if (j == 2) {
						interp =
							ue_interpolation_pf2[predictor_index][
								interpolation_index];
					}
#else // STANDALONE
					if (j == 0) {
						interp =
							ue->ue_interpolation_pf0[predictor_index][
								interpolation_index];
					}
					else if (j == 1) {
						interp =
							ue->ue_interpolation_pf1[predictor_index][
								interpolation_index];
					}
					else if (j == 2) {
						interp =
							ue->ue_interpolation_pf2[predictor_index][
								interpolation_index];
					}
#endif // STANDALONE
					// Interpolate down to the finely binned
					// pseudorapidity

					density += pred /
						(2.0 * M_PI *
						 (_edge_pseudorapidity[predictor_index + 1] -
						  _edge_pseudorapidity[predictor_index])) *
						interp;
				}
				}

				if (std::isfinite(iterator->area) && density >= 0) {
					// Subtract the PF candidate by density times
					// Voronoi cell area
					iterator->momentum_perp_subtracted =
						iterator->momentum.Pt() -
						density * iterator->area;
				}
				else {
					iterator->momentum_perp_subtracted =
						iterator->momentum.Pt();
				}
				iterator->momentum_perp_subtracted_unequalized =
					iterator->momentum_perp_subtracted;
			}
		}
