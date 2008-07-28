ConfDB::getConfigId(fullConfigName=/dev/CMSSW_2_1_0_pre10/HLT) failed (dirName=/dev/CMSSW_2_1_0_pre10, configName=HLT,version=2): Exhausted Resultset
cause: Exhausted Resultset

confdb.db.DatabaseException: ConfDB::getConfigId(fullConfigName=/dev/CMSSW_2_1_0_pre10/HLT) failed (dirName=/dev/CMSSW_2_1_0_pre10, configName=HLT,version=2): Exhausted Resultset
	at confdb.db.ConfDB.getConfigId(ConfDB.java:1946)
	at org.apache.jsp.get_jsp._jspService(get_jsp.java:143)
	at org.apache.jasper.runtime.HttpJspBase.service(HttpJspBase.java:98)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:803)
	at org.apache.jasper.servlet.JspServletWrapper.service(JspServletWrapper.java:328)
	at org.apache.jasper.servlet.JspServlet.serviceJspFile(JspServlet.java:315)
	at org.apache.jasper.servlet.JspServlet.service(JspServlet.java:265)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:803)
	at sun.reflect.GeneratedMethodAccessor952.invoke(Unknown Source)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:25)
	at java.lang.reflect.Method.invoke(Method.java:585)
	at org.apache.catalina.security.SecurityUtil$1.run(SecurityUtil.java:244)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAsPrivileged(Subject.java:517)
	at org.apache.catalina.security.SecurityUtil.execute(SecurityUtil.java:276)
	at org.apache.catalina.security.SecurityUtil.doAsPrivilege(SecurityUtil.java:162)
	at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:262)
	at org.apache.catalina.core.ApplicationFilterChain.access$000(ApplicationFilterChain.java:52)
	at org.apache.catalina.core.ApplicationFilterChain$1.run(ApplicationFilterChain.java:171)
	at java.security.AccessController.doPrivileged(Native Method)
	at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:167)
	at org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:210)
	at org.apache.catalina.core.StandardContextValve.invoke(StandardContextValve.java:174)
	at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:127)
	at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:117)
	at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:108)
	at org.apache.catalina.connector.CoyoteAdapter.service(CoyoteAdapter.java:151)
	at org.apache.jk.server.JkCoyoteHandler.invoke(JkCoyoteHandler.java:200)
	at org.apache.jk.common.HandlerRequest.invoke(HandlerRequest.java:283)
	at org.apache.jk.common.ChannelSocket.invoke(ChannelSocket.java:773)
	at org.apache.jk.common.ChannelSocket.processConnection(ChannelSocket.java:703)
	at org.apache.jk.common.ChannelSocket$SocketConnection.runIt(ChannelSocket.java:895)
	at org.apache.tomcat.util.threads.ThreadPool$ControlRunnable.run(ThreadPool.java:685)
	at java.lang.Thread.run(Thread.java:595)
Caused by: java.sql.SQLException: Exhausted Resultset
	at oracle.jdbc.driver.DatabaseError.throwSqlException(DatabaseError.java:112)
	at oracle.jdbc.driver.DatabaseError.throwSqlException(DatabaseError.java:146)
	at oracle.jdbc.driver.DatabaseError.throwSqlException(DatabaseError.java:208)
	at oracle.jdbc.driver.OracleResultSetImpl.getInt(OracleResultSetImpl.java:506)
	at confdb.db.ConfDB.getConfigId(ConfDB.java:1939)
	... 33 more




