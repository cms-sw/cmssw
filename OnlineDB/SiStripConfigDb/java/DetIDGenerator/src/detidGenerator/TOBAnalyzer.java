package detidGenerator;

import db.*;
import java.util.Vector;

public class TOBAnalyzer implements IDetIdGenerator{

    Vector<Vector<String>> detIds;
    CDBConnection c;

    /**
       @todo fill detIds with [object_id, detId] for the TOB
    **/
    public TOBAnalyzer() throws java.sql.SQLException, java.lang.ClassNotFoundException{
	detIds = new Vector<Vector<String>>();
	c = CDBConnection.getConnection();

    }

    public Vector<Vector<String>> getDetIds(){
	return detIds;
    }

}