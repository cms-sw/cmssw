package detidGenerator;
import java.util.ArrayList;

/**
 * @author G. Baulieu
 * @version 1.0
 **/

/*
  $Date: 2007/01/18 17:04:45 $
  
  $Log: IDetIdGenerator.java,v $
  Revision 1.2  2007/01/18 17:04:45  gbaulieu
  Use an external library for database connections
  Use preparedStatements to speed up the queries

  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.2  2006/02/02 17:17:00  baulieu
  Some modifications for JDK 1.5
  Call a PL/SQL function to export the parameters


*/

public interface IDetIdGenerator{
    /**
       Returns a Vector of Vectors containing the (object_id, det_id) couples : [[object_id1, det_id1], [object_id2, det_id2]]
    **/
    ArrayList<ArrayList<String>> getDetIds();
}