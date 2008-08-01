#include "RecoParticleFlow/Benchmark/interface/histo.h"
#include <string>
#include <iostream>

using namespace std;

histo::histo()
{
	//this->name_=new string();
}

histo::~histo()
{

}

void histo::setname(string name)
{
	this->name_=new string(name);
	//cout<<"Name: "<<(*this->name_)<<endl;
}

void histo::setbins(int bins)
{
	this->bins_=bins;
	//cout<<"Bins: "<<this->bins_<<endl;
}

void histo::setmin(int min)
{
	this->min_=min;
	//cout<<"Min: "<<this->min_<<endl;
}

void histo::setmax(int max)
{
	this->max_=max;
	//cout<<"Max: "<<this->max_<<endl;
}

string* histo::getname()
{
	return this->name_;
}

int histo::getbins()
{
	return this->bins_;
}

int histo::getmin()
{
	return this->min_;
}

int histo::getmax()
{
	return this->max_;
}
