# Templates for the production of a LaTeX presentation.

texTemplate=r"""%Offline Alignment Validation presentation.
%Time of creation: [time]
%Created with produceOfflineValidationTex.py
\documentclass{beamer}
\usepackage[latin1]{inputenc}
\usepackage{color}
%\usepackage{siunitx}
%\usepackage{epstopdf}
\usetheme{default}
\title[Offline Validation]{Title here}
\author{Author(s) here}
\institute{Institute here}
\date{Date here}



\begin{document}

\begin{frame}
\titlepage
\end{frame}

\section{Introduction}
%---------------------------------------------
\begin{frame}{Introduction}
{
\begin{itemize}
 \item Introduction here
\end{itemize}
}
\end{frame}


\section{Plots}
%---------------------------------------------


[frames]


\section{Conclusions}
%---------------------------------------------
\begin{frame}{Conclusions}
{
\begin{itemize}
\item Conclusions here
\end{itemize}
}
\end{frame} 



\end{document}

"""

frameTemplate=r"""
\begin{frame}{[title]}
  \begin{figure}
    \centering
[plots]
    %\\Comments here
  \end{figure}
\end{frame}
"""

plotTemplate=r"""    \includegraphics[width=[width]\textwidth, height=[height]\textheight, keepaspectratio=true]{[path]}"""

subsectionTemplate=r"""

\subsection{[title]}
%---------------------------------------------
"""

toPdf="""
#To produce a pdf presentation 
#a. fill in your information, comments etc. in presentation.tex
#b. run this script: ./ToPdf.sh
latex presentation.tex
latex presentation.tex #(twice to produce the bookmarks)
dvipdf presentation.dvi
#(pdflatex doesn't like .eps-images; this way we can
#use just latex and the convert the result into pdf.)

"""
