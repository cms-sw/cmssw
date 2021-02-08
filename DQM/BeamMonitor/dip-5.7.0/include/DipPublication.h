#ifndef DIPPUBLICATION_H_INCLUDED
#define DIPPUBLICATION_H_INCLUDED

#include "DipQuality.h"
#include "DipTimestamp.h"
#include "DipData.h"
#include "DipException.h"
#include "StdTypes.h"
#include <string>

class DipDllExp DipPublication {
protected:
	DipPublication() { }
	virtual ~DipPublication() { }

public:
	virtual const char* getTopicName() const = 0;


	virtual void send(const DipData& value,		const DipTimestamp & timestamp) = 0;
    virtual void send(const DipData &value,     const DipTimestamp & timestamp, DipQuality quality, const char * qualityReason) = 0;

	virtual void send(const char *value,		const DipTimestamp & timestamp) = 0;
	virtual void send(const std::string &value, const DipTimestamp & timestamp) = 0;
	virtual void send(DipBool value,			const DipTimestamp & timestamp) = 0;
	virtual void send(DipByte value,			const DipTimestamp & timestamp) = 0;
	virtual void send(DipShort value,			const DipTimestamp & timestamp) = 0;
	virtual void send(DipInt value,				const DipTimestamp & timestamp) = 0;
	virtual void send(DipLong value,			const DipTimestamp & timestamp) = 0;
	virtual void send(DipFloat value,			const DipTimestamp & timestamp) = 0;
	virtual void send(DipDouble value,			const DipTimestamp & timestamp) = 0;
	virtual void send(const char **value,		int size,   const DipTimestamp & timestamp) = 0;
	virtual void send(const std::string *value, int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipBool *value,		int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipByte *value,		int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipShort *value,	int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipInt *value,		int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipLong *value,		int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipFloat *value,	int size,	const DipTimestamp & timestamp) = 0;
	virtual void send(const DipDouble *value,	int size,	const DipTimestamp & timestamp) = 0;
	virtual void setQualityBad() = 0;
	virtual void setQualityBad(const char * reason) = 0;
	virtual void setQualityUncertain() = 0;
	virtual void setQualityUncertain(const char *reason) = 0;
};

#endif //DIPPUBLICATION_H_INCLUDED
