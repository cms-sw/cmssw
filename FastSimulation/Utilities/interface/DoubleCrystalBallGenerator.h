#ifndef DoubleCrystalBallGenerator_H
#define DoubleCrystalBallGenerator_H

/**
 * This class provides a Crystal Ball function generator
 * The power-law tail can be either on the right side (default) or the left side
 * \author Kevin Pedro
 * $Date: 14 March 2013 */

class RandomEngine;

class DoubleCrystalBallGenerator
{
	public:
		//constructor
		DoubleCrystalBallGenerator(const RandomEngine* engine) : random(engine) {}
		//destructor
		virtual ~DoubleCrystalBallGenerator() {}
		
		//functions
		double shoot(double mu, double sigma, double aL, double nL, double aR, double nR);

	private:
		//random engine
		const RandomEngine* random;

};

#endif