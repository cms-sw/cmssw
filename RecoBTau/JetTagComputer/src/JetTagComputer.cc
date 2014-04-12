#include <vector>
#include <string>

#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

float JetTagComputer::operator () (const reco::BaseTagInfo& info) const
{
	std::vector<const reco::BaseTagInfo*> helper;
	helper.push_back(&info);
	return discriminator(TagInfoHelper(helper));
}

void JetTagComputer::uses(unsigned int id, const std::string &label)
{
	if (m_setupDone)
		throw cms::Exception("InvalidState")
			<< "JetTagComputer::uses called outside of "
			   "constructor" << std::endl;

	if (id >= m_inputLabels.size())
		m_inputLabels.resize(id + 1);

	if (!m_inputLabels[id].empty())
		throw cms::Exception("InvalidIndex")
			<< "JetTagComputer::uses called with duplicate "
			   "or invalid index" << std::endl;

	m_inputLabels[id] = label;
}

float JetTagComputer::discriminator(const reco::BaseTagInfo &info) const
{
	throw cms::Exception("VirtualMethodMissing")
		<< "No virtual method implementation for "
		<< "JetTagComputer::discriminator() defined." << std::endl;
}

float JetTagComputer::discriminator(const JetTagComputer::TagInfoHelper &info) const
{
	return discriminator(info.getBase(0));
}

TYPELOOKUP_DATA_REG(JetTagComputer);
