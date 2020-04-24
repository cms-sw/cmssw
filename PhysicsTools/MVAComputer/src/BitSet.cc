#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

namespace PhysicsTools {

BitSet::size_t BitSet::bits() const
{
	static constexpr unsigned char byteBits[256] = {
		0,  1,  1,  2,  1,  2,  2,  3,  1,  2,  2,  3,  2,  3,  3,    
		4,  1,  2,  2,  3,  2,  3,  3,  4,  2,  3,  3,  4,  3,  4,
		4,  5,  1,  2,  2,  3,  2,  3,  3,  4,  2,  3,  3,  4,  3,    
		4,  4,  5,  2,  3,  3,  4,  3,  4,  4,  5,  3,  4,  4,  5,    
		4,  5,  5,  6,  1,  2,  2,  3,  2,  3,  3,  4,  2,  3,  3,    
		4,  3,  4,  4,  5,  2,  3,  3,  4,  3,  4,  4,  5,  3,  4,
		4,  5,  4,  5,  5,  6,  2,  3,  3,  4,  3,  4,  4,  5,  3,
		4,  4,  5,  4,  5,  5,  6,  3,  4,  4,  5,  4,  5,  5,  6,
		4,  5,  5,  6,  5,  6,  6,  7,  1,  2,  2,  3,  2,  3,  3,    
		4,  2,  3,  3,  4,  3,  4,  4,  5,  2,  3,  3,  4,  3,  4,    
		4,  5,  3,  4,  4,  5,  4,  5,  5,  6,  2,  3,  3,  4,  3,    
		4,  4,  5,  3,  4,  4,  5,  4,  5,  5,  6,  3,  4,  4,  5,    
		4,  5,  5,  6,  4,  5,  5,  6,  5,  6,  6,  7,  2,  3,  3,
		4,  3,  4,  4,  5,  3,  4,  4,  5,  4,  5,  5,  6,  3,  4,    
		4,  5,  4,  5,  5,  6,  4,  5,  5,  6,  5,  6,  6,  7,  3,    
		4,  4,  5,  4,  5,  5,  6,  4,  5,  5,  6,  5,  6,  6,  7,
		4,  5,  5,  6,  5,  6,  6,  7,  5,  6,  6,  7,  6,  7,  7,    
		8
	};
	const unsigned char *begin = reinterpret_cast<const unsigned char*>(store);
	const unsigned char *end   = reinterpret_cast<const unsigned char*>(store + bitsToWords(bits_));

	size_t bits = 0;
	for(const unsigned char *p = begin; p < end; p++)
		bits += byteBits[*p];

	return bits;
}

BitSet Calibration::convert(const Calibration::BitSet &bitSet)
{
	PhysicsTools::BitSet::size_t size = bitSet.store.size();
	size = (size - 1) * 8 + (bitSet.bitsInLast + 7) % 8 + 1;

	PhysicsTools::BitSet result(size);
	for(PhysicsTools::BitSet::size_t i = 0; i < size; i++)
		result[i] = bitSet.store[i / 8] & (1 << (i % 8));

	return result;
}

Calibration::BitSet Calibration::convert(const PhysicsTools::BitSet &bitSet)
{
	PhysicsTools::BitSet::size_t size = bitSet.size();
	PhysicsTools::BitSet::size_t bytes = (size + 7) / 8;

	Calibration::BitSet result;
	result.store.resize(bytes);
	result.bitsInLast = (size + 7) % 8 + 1;

	for(PhysicsTools::BitSet::size_t i = 0; i < size; i++)
		result.store[i / 8] |= bitSet[i] ? (1 << (i % 8)) : 0;

	return result;
}

} // namespace PhysicsTools
