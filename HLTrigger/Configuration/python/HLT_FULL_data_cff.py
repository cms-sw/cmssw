null

java.lang.NullPointerException
	at confdb.db.ConfDB.getConfigId(ConfDB.java:2806)
	at org.apache.jsp.get_jsp._jspService(get_jsp.java:95)
	at org.apache.jasper.runtime.HttpJspBase.service(HttpJspBase.java:70)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:717)
	at org.apache.jasper.servlet.JspServletWrapper.service(JspServletWrapper.java:374)
	at org.apache.jasper.servlet.JspServlet.serviceJspFile(JspServlet.java:342)
	at org.apache.jasper.servlet.JspServlet.service(JspServlet.java:267)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:717)
	at sun.reflect.GeneratedMethodAccessor56.invoke(Unknown Source)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:25)
	at java.lang.reflect.Method.invoke(Method.java:597)
	at org.apache.catalina.security.SecurityUtil$1.run(SecurityUtil.java:244)
	at java.security.AccessController.doPrivileged(Native Method)
	at javax.security.auth.Subject.doAsPrivileged(Subject.java:517)
	at org.apache.catalina.security.SecurityUtil.execute(SecurityUtil.java:276)
	at org.apache.catalina.security.SecurityUtil.doAsPrivilege(SecurityUtil.java:162)
	at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:283)
	at org.apache.catalina.core.ApplicationFilterChain.access$000(ApplicationFilterChain.java:56)
	at org.apache.catalina.core.ApplicationFilterChain$1.run(ApplicationFilterChain.java:189)
	at java.security.AccessController.doPrivileged(Native Method)
	at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:185)
	at org.apache.catalina.core.StandardWrapperValve.invoke(StandardWrapperValve.java:233)
	at org.apache.catalina.core.StandardContextValve.invoke(StandardContextValve.java:191)
	at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:128)
	at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:102)
	at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:109)
	at org.apache.catalina.connector.CoyoteAdapter.service(CoyoteAdapter.java:286)
	at org.apache.jk.server.JkCoyoteHandler.invoke(JkCoyoteHandler.java:190)
	at org.apache.jk.common.HandlerRequest.invoke(HandlerRequest.java:283)
	at org.apache.jk.common.ChannelSocket.invoke(ChannelSocket.java:767)
	at org.apache.jk.common.ChannelSocket.processConnection(ChannelSocket.java:697)
	at org.apache.jk.common.ChannelSocket$SocketConnection.runIt(ChannelSocket.java:889)
	at org.apache.tomcat.util.threads.ThreadPool$ControlRunnable.run(ThreadPool.java:690)
	at java.lang.Thread.run(Thread.java:619)





# version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

