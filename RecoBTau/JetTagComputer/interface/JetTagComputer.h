#ifndef RecoBTau_JetTagComputer_h
#define RecoBTau_JetTagComputer_h

#include <vector>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

class JetTagComputer {
    public:
	class TagInfoHelper {
	    public:
		TagInfoHelper(const std::vector<const reco::BaseTagInfo*> &infos) :
			m_tagInfos(infos) {}
		~TagInfoHelper() {}

		const reco::BaseTagInfo &getBase(unsigned int index) const
		{
			if (index >= m_tagInfos.size())
				throw cms::Exception("InvalidIndex")
					<< "Invalid index " << index << " "
					   "in call to JetTagComputer::get."
					<< std::endl;

			const reco::BaseTagInfo *info = m_tagInfos[index];
			if (!info)
				throw cms::Exception("ProductMissing")
					<< "Missing TagInfo "
					   "in call to JetTagComputer::get."
					<< std::endl;

			return *info;
		}

		template<class T>
		const T &get(unsigned int index = 0) const
		{
			const reco::BaseTagInfo *info = &getBase(index);
			const T *castInfo = dynamic_cast<const T*>(info);
			if (!castInfo)
				throw cms::Exception("InvalidCast")
					<< "Invalid TagInfo cast "
					   "in call to JetTagComputer::get."
					<< std::endl;

			return *castInfo;
		}

	    private:
		const std::vector<const reco::BaseTagInfo*>	&m_tagInfos;
	};

	// default constructor
	JetTagComputer() : m_setupDone(false) {}
	virtual ~JetTagComputer() {}

	// explicit constructor accepting a ParameterSet for configuration
	explicit JetTagComputer(const edm::ParameterSet& configuration) :
		m_setupDone(false) {}

	virtual void setEventSetup(const edm::EventSetup&) const {}

	float operator () (const reco::BaseTagInfo& info) const;
	inline float operator () (const TagInfoHelper &helper) const
	{ return discriminator(helper); }

	inline const std::vector<std::string> &getInputLabels() const
	{ m_setupDone = true; return m_inputLabels; }

    protected:
	void uses(unsigned int id, const std::string &label);
	void uses(const std::string &label) { uses(0, label); }

	virtual float discriminator(const reco::BaseTagInfo&) const;
	virtual float discriminator(const TagInfoHelper&) const;

    private:
	std::vector<std::string>	m_inputLabels;
	mutable bool			m_setupDone;
};

#endif // RecoBTau_JetTagComputer_h
