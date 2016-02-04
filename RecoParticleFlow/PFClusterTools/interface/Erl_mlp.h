#ifndef ERL_MLP_H_
#define ERL_MLP_H_
#include <cmath>
#include <iostream>
namespace pftools {

#define THISALGORITHMBECOMINGSKYNETCOST = 9999;

class Erl_mlp {
public:
	Erl_mlp();

	virtual ~Erl_mlp();

	void setOffsetAndSlope(const double offset, const double slope) {
		offset_ = offset;
		slope_ = slope;
	}

	double evaluate(const double t1 = 0.0, const double t2 = 0.0,
			const double t3 = 0.0, const double t4 = 0.0,
			const double t5 = 0.0, const double t6 = 0.0, const double t7 = 0.0) {
		t1_ = t1;
		t2_ = t2;
		t3_ = t3;
		t4_ = t4;
		t5_ = t5;
		t6_ = t6;
		t7_ = t7;

		return output();
	}

	double ecalFraction(const double t1 = 0.0, const double t2 = 0.0,
			const double t3 = 0.0, const double t4 = 0.0,
			const double t5 = 0.0, const double t6 = 0.0, const double t7 = 0.0) {
		t1_ = t1;
		t2_ = t2;
		t3_ = t3;
		t4_ = t4;
		t5_ = t5;
		t6_ = t6;
		t7_ = t7;

		return ecalOutput();
	}

private:
	double t1_, t2_, t3_, t4_, t5_, t6_, t7_;
	double offset_, slope_;

	inline double transform(const double k) {
		return tanh(k);
		//return k/2.0;
	}

	double output() {
		return ((80 * neuron_01190()) - offset_) / slope_;
		//return 0;
	}

	double ecalOutput() {
		return neuron_01200() * 8.0/5.0;
	}
                                                        
	double neuron_01190() { return 5.06272330366805203838e-02 * neuron_01110() + 7.27249962994314103071e-02 * neuron_01100() + 6.87531379353593874448e-01 * neuron_01090() + 5.90999649535358523300e-02 * neuron_01170() + 5.11852363238117641364e-02 * neuron_01160() + 4.49943694412880193512e-02 * neuron_01150() + 4.07425660324388025368e-02 * neuron_01140() + 3.93358734274708785050e-02 * neuron_01130() + 4.19854685691317813800e-02 * neuron_01120(); }                                                                                                                                 
	double neuron_01200() { return 2.45908500608800761889e-01 * neuron_01110() + 2.87953657915926142241e-01 * neuron_01100() + 1.12104946100116772967e-01 * neuron_01090() + -2.48512814545735039040e-01 * neuron_01170() + -1.73640293653000848950e-01 * neuron_01160() + -9.57729315919204349239e-02 * neuron_01150() + -1.34634120232926899480e-02 * neuron_01140() + 7.33600499895466356959e-02 * neuron_01130() + 1.62882613167782547281e-01 * neuron_01120(); }                                                                                                                             
																																																	
	//Hidden layer: <0.118.0>                                                                                                                                                                       
	double neuron_01090() { return transform(1.77006431112942702599e-01 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + 2.14497710612496367277e-01 * neuron_01050() + 1.43625191299545074131e+00 * neuron_01040() + 1.35895192827592214968e+00 * neuron_01030() + 3.42315251134937381661e-01 * neuron_01020() + 2.12812599023514514851e-01 * neuron_01010()) ; }                 
	double neuron_01100() { return transform(2.43335108048174653117e-02 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -1.54265303752188047415e-02 * neuron_01050() + 5.31887393676174879964e-01 * neuron_01040() + 1.29700229375154463263e+00 * neuron_01030() + -2.51855201382369053853e-01 * neuron_01020() + -5.06439475438290737097e-02 * neuron_01010()) ; }              
	double neuron_01110() { return transform(2.26240107787991917565e-02 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -5.41830867221511913723e-02 * neuron_01050() + 5.34848607670024622784e-01 * neuron_01040() + 1.31149902444579002925e+00 * neuron_01030() + -2.64338982863860205708e-01 * neuron_01020() + -4.06486696831378502281e-02 * neuron_01010()) ; }              
	double neuron_01120() { return transform(3.53971299195493110945e-02 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -8.37804537629018991618e-02 * neuron_01050() + 6.07782495555854840319e-01 * neuron_01040() + 1.18563130862724230852e+00 * neuron_01030() + -2.68121286212209153366e-01 * neuron_01020() + -6.12015780590782140780e-02 * neuron_01010()) ; }              
	double neuron_01130() { return transform(5.69937856406597387338e-02 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -1.00016140981091974926e-01 * neuron_01050() + 7.03662421839865714901e-01 * neuron_01040() + 9.52968317012650611986e-01 * neuron_01030() + -2.73076926608740555569e-01 * neuron_01020() + -7.68552378252462947694e-02 * neuron_01010()) ; }              
	double neuron_01140() { return transform(8.30357728126230598686e-02 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -1.03499741661745067733e-01 * neuron_01050() + 8.06348739275546733118e-01 * neuron_01040() + 6.38758549749190884803e-01 * neuron_01030() + -2.80508796980343866334e-01 * neuron_01020() + -7.87991143999155069233e-02 * neuron_01010()) ; }              
	double neuron_01150() { return transform(1.09487813610100057082e-01 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -9.70768580095677652286e-02 * neuron_01050() + 9.10591921032660733815e-01 * neuron_01040() + 2.66241721261271546695e-01 * neuron_01030() + -2.89724454229762784507e-01 * neuron_01020() + -6.80058450136514086592e-02 * neuron_01010()) ; }              
	double neuron_01160() { return transform(1.32909925381862498162e-01 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -8.45644764303960794205e-02 * neuron_01050() + 1.01870729141489779757e+00 * neuron_01040() + -1.39709767812830482070e-01 * neuron_01030() + -3.01085409364046208136e-01 * neuron_01020() + -4.85758323472233188856e-02 * neuron_01010()) ; }             
	double neuron_01170() { return transform(1.50514730336528679278e-01 * neuron_01070() + 2.00000000000000011102e-01 * neuron_01060() + -7.16821794819610530469e-02 * neuron_01050() + 1.14442797677888430385e+00 * neuron_01040() + -5.53609997511943840998e-01 * neuron_01030() + -3.18164032162814081062e-01 * neuron_01020() + -2.52943742434275434250e-02 * neuron_01010()) ; }             
																																																	
	//Input layer: <0.108.0>                                                                                                                                                                        
	double neuron_01010() { return t1_; }
	double neuron_01020() { return t2_; }
	double neuron_01030() { return t3_; }
	double neuron_01040() { return t4_; }
	double neuron_01050() { return t5_; }
	double neuron_01060() { return t6_; }
	double neuron_01070() { return t7_; }
};
}

#endif /*ERL_MLP_H_*/
