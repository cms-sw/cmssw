#ifndef __HiJetAlgos_UEParameters_h__
#define __HiJetAlgos_UEParameters_h__

#include <boost/multi_array.hpp>

class UEParameters {

private:
	static const size_t nreduced_particle_flow_id = 3;
	const std::vector<float> *v_;
	int nn_;
	int neta_;
	boost::const_multi_array_ref<float, 4> *parameters_;

public:
	UEParameters(const std::vector<float> *v, int nn, int neta);
        ~UEParameters(){delete parameters_;}
	const std::vector<float>& get_raw(void) const {return *v_;}
	
	void get_fourier(double &re, double &im, size_t n, size_t eta, int type = -1) const;
	double get_sum_pt(int eta, int type = -1) const;
	double get_vn(int n, int eta, int type = -1) const;
	double get_psin(int n, int eta, int type = -1) const;
};

#endif
