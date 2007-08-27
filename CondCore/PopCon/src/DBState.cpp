#include "CondCore/PopCon/interface/DBState.h"

bool popcon::operator==(const DBState& a, const DBState& b)
{
	//return ::memcmp(&a, &b, sizeof(popcon::DBState))==0; 
	//TODO compare only the size
	return a.payload_size == b.payload_size ? true : false;
}

popcon::DBState::DBState (std::string oname, std::string sname, int ps ) : name(oname), schema(sname), payload_size(ps) {}


popcon::DBState::DBState (coral::AttributeList lst) {
	//name = lst["N"].data<std::string>();
	//schema = lst["S"].data<std::string>();
	//payload_table = lst["PT"].data<std::string>();
	//payload_size = lst["PS"].data<std::string>();
	//last_iov = lst["LIOV"].data<int>();
	//except_description = lst["ED"].data<std::string>();
	//manual_override = lst["MO"].data<std::string>();
	name = lst[0].data<std::string>();
	schema = lst[1].data<std::string>();
	payload_size = lst[2].data<int>();
	except_description = lst[3].data<std::string>();
	manual_override = lst[4].data<std::string>();
}

void popcon::DBState::set_state (coral::AttributeList lst){
	//name = lst["N"].data<std::string>();
	//schema = lst["S"].data<std::string>();
	//payload_table = lst["PT"].data<std::string>();
	//payload_size = lst["PS"].data<std::string>();
	//last_iov = lst["LIOV"].data<int>();
	//except_description = lst["ED"].data<std::string>();
	//manual_override = lst["MO"].data<std::string>();
	name = lst[0].data<std::string>();
	schema = lst[1].data<std::string>();
	payload_size = lst[2].data<int>();
	except_description = lst[3].data<std::string>();
	manual_override = lst[4].data<std::string>();
}


coral::AttributeList popcon::DBState::update_helper(std::string& updateAction, std::string& updateCondition)
{	

	//FIXME impossible to update with join!
	coral::AttributeList udata;
	udata.extend<int>("PS");
	udata.extend<std::string>("ED");
	udata.extend<std::string>("MO");
	udata.extend<std::string>("NM");
	udata[0].data<int>() = payload_size;
	udata[1].data<std::string>() = except_description;
	udata[2].data<std::string>() = manual_override;
	udata[3].data<std::string>() = name;

	updateCondition = "NAME = :NM";
	updateAction = "PAYLOAD_SIZE = :PS, EXCEPT_DESCRIPTION = :ED, MANUAL_OVERRIDE = :MO";
	return udata;

}
