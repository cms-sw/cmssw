#ifndef __private_MLP_h
#define __private_MLP_h
#include <string>
namespace PhysicsTools {

class MLP {
    public:
	MLP(unsigned int nIn, unsigned int nOut, const std::string layout);
	~MLP();

	void clear();
	void init(unsigned int rows);
	void set(unsigned int row, double *data, double *target, double weight = 1.0);
	double train();
	const double *eval(double *data) const;
	void save(const std::string file) const;
	void load(const std::string file);

	inline unsigned int getEpoch() const { return epoch; }
	inline int getLayers() const { return layers; }
	inline const int *getLayout() const { return layout; }

    private:
	void		setLearn(void);
	void		setNPattern(unsigned int size);

	bool		initialized;
	int		layers;
	int		*layout;

	unsigned int	epoch;
	static bool	inUse;
};

} // namespace PhysicsTools

#endif // _private_MLP_h
