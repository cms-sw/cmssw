package db;

/**
 * <p>Sent when a class is used where it should not ...</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
 $Date: 2006/06/28 11:42:24 $

 $Log: ClassNotSupportedException.java,v $
 Revision 1.1  2006/06/28 11:42:24  gbaulieu
 First import of the sources

 Revision 1.2  2006/06/07 12:40:42  baulieu
 Add a - verbose option
 Add a serialVersionUID to the ClassNotSupportedException class to avoid a warning

 Revision 1.1  2006/02/02 17:17:00  baulieu
 Some modifications for JDK 1.5
 Call a PL/SQL function to export the parameters


*/

public class ClassNotSupportedException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public ClassNotSupportedException(String message) {
    	super(message);
    }

}
