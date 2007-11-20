#include "CondCore/PopCon/interface/DBState.h"

bool popcon::operator==(const DBState& a, const DBState& b)
{
	return a.payload_size == b.payload_size ? true : false;
}

popcon::DBState::DBState (const std::string& oname, 
			  int ps, 
			  const std::string& cs ) : name(oname),  payload_size(ps), connect_string(cs) {}


popcon::DBState::DBState (coral::AttributeList& lst, const std::string& cs) {
	name = lst[0].data<std::string>();
	payload_size = lst[1].data<int>();
	except_description = lst[2].data<std::string>();
	manual_override = lst[3].data<std::string>();
	connect_string = cs;
}

void popcon::DBState::set_state (const coral::AttributeList& lst){
	name = lst[0].data<std::string>();
	payload_size = lst[1].data<int>();
	except_description = lst[2].data<std::string>();
	manual_override = lst[3].data<std::string>();
}


coral::AttributeList popcon::DBState::update_helper(std::string& updateAction, std::string& updateCondition)
{	

	coral::AttributeList udata;
	udata.extend<int>("PS");
	udata.extend<std::string>("ED");
	udata.extend<std::string>("MO");
	udata.extend<std::string>("NM");
	udata.extend<std::string>("CS");
	udata[0].data<int>() = payload_size;
	udata[1].data<std::string>() = except_description;
	udata[2].data<std::string>() = manual_override;
	udata[3].data<std::string>() = name;
	udata[4].data<std::string>() = connect_string;

	updateCondition = "NAME = :NM and CONNECT_STRING = :CS";
	updateAction = "PAYLOAD_SIZE = :PS, EXCEPT_DESCRIPTION = :ED, MANUAL_OVERRIDE = :MO";
	return udata;

}
