package detidGenerator;
import java.util.Vector;

/**
 * @author G. Baulieu
 * @version 1.0
 **/

/*
  $Date: 2006/02/02 17:17:00 $
  
  $Log: IDetIdGenerator.java,v $
  Revision 1.2  2006/02/02 17:17:00  baulieu
  Some modifications for JDK 1.5
  Call a PL/SQL function to export the parameters


*/

public interface IDetIdGenerator{
    /**
       Returns a Vector of Vectors containing the (object_id, det_id) couples : [[object_id1, det_id1], [object_id2, det_id2]]
    **/
    Vector<Vector<String>> getDetIds();
}