webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/workspaces/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];












var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__["store"]),
      workspace = _React$useContext.workspace,
      setWorkspace = _React$useContext.setWorkspace;

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var initialWorkspace = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? workspaces[0].workspaces[1].label : workspaces[0].workspaces[3].label;
  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(initialWorkspace);
    return function () {
      return setWorkspace(initialWorkspace);
    };
  }, []);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1]; // make a workspace set from context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_12__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleWorkspaces(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 58,
        columnNumber: 13
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 70,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 72,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "9wsb3E7mFlyFmQpi1Uvfk2BcVak=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJSZWFjdCIsInN0b3JlIiwid29ya3NwYWNlIiwic2V0V29ya3NwYWNlIiwid29ya3NwYWNlcyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwib25saW5lV29ya3NwYWNlIiwib2ZmbGluZVdvcnNrcGFjZSIsImluaXRpYWxXb3Jrc3BhY2UiLCJsYWJlbCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5Iiwib3BlbldvcmtzcGFjZXMiLCJ0b2dnbGVXb3Jrc3BhY2VzIiwidGhlbWUiLCJjb2xvcnMiLCJzZWNvbmRhcnkiLCJtYWluIiwibWFwIiwic3ViV29ya3NwYWNlIiwic2V0V29ya3NwYWNlVG9RdWVyeSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBR0E7QUFDQTtBQUNBO0lBRVFBLE8sR0FBWUMseUMsQ0FBWkQsTzs7QUFNUixJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQUEsMEJBQ2FDLGdEQUFBLENBQWlCQyxnRUFBakIsQ0FEYjtBQUFBLE1BQ2ZDLFNBRGUscUJBQ2ZBLFNBRGU7QUFBQSxNQUNKQyxZQURJLHFCQUNKQSxZQURJOztBQUd2QixNQUFNQyxVQUFVLEdBQ2RDLGdFQUFnQixDQUFDQyxJQUFqQixLQUEwQixRQUExQixHQUFxQ0MsNkRBQXJDLEdBQXVEQyw4REFEekQ7QUFHQSxNQUFNQyxnQkFBZ0IsR0FBR0osZ0VBQWdCLENBQUNDLElBQWpCLEtBQTBCLFFBQTFCLEdBQXFDRixVQUFVLENBQUMsQ0FBRCxDQUFWLENBQWNBLFVBQWQsQ0FBeUIsQ0FBekIsRUFBNEJNLEtBQWpFLEdBQXlFTixVQUFVLENBQUMsQ0FB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S0FiWGIsVTtBQXdFU0EseUVBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDZhMWQzZGI5MGYxMzkxYjI1ZGUuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyB3b3Jrc3BhY2VzIGFzIG9mZmxpbmVXb3Jza3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29mZmxpbmUnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XG5pbXBvcnQgeyBTdHlsZWRNb2RhbCB9IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCBGb3JtIGZyb20gJ2FudGQvbGliL2Zvcm0vRm9ybSc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBzZXRXb3Jrc3BhY2VUb1F1ZXJ5IH0gZnJvbSAnLi91dGlscyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdXNlQ2hhbmdlUm91dGVyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlQ2hhbmdlUm91dGVyJztcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vc3R5bGVzL3RoZW1lJztcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcblxuY29uc3QgeyBUYWJQYW5lIH0gPSBUYWJzO1xuXG5pbnRlcmZhY2UgV29yc3BhY2VQcm9wcyB7XG4gIGxhYmVsOiBzdHJpbmc7XG4gIHdvcmtzcGFjZXM6IGFueTtcbn1cbmNvbnN0IFdvcmtzcGFjZXMgPSAoKSA9PiB7XG4gIGNvbnN0IHsgd29ya3NwYWNlLCBzZXRXb3Jrc3BhY2UgfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpXG5cbiAgY29uc3Qgd29ya3NwYWNlcyA9XG4gICAgZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJyA/IG9ubGluZVdvcmtzcGFjZSA6IG9mZmxpbmVXb3Jza3BhY2U7XG4gICAgXG4gIGNvbnN0IGluaXRpYWxXb3Jrc3BhY2UgPSBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzFdLmxhYmVsIDogd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzNdLmxhYmVsXG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcbiAgICByZXR1cm4gKCkgPT4gc2V0V29ya3NwYWNlKGluaXRpYWxXb3Jrc3BhY2UpXG4gIH0sIFtdKVxuXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCBbb3BlbldvcmtzcGFjZXMsIHRvZ2dsZVdvcmtzcGFjZXNdID0gUmVhY3QudXNlU3RhdGUoZmFsc2UpO1xuXG4gIC8vIG1ha2UgYSB3b3Jrc3BhY2Ugc2V0IGZyb20gY29udGV4dFxuICByZXR1cm4gKFxuICAgIDxGb3JtPlxuICAgICAgPFN0eWxlZEZvcm1JdGVtIGxhYmVsY29sb3I9XCJ3aGl0ZVwiIGxhYmVsPVwiV29ya3NwYWNlXCI+XG4gICAgICAgIDxCdXR0b25cbiAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XG4gICAgICAgICAgICB0b2dnbGVXb3Jrc3BhY2VzKCFvcGVuV29ya3NwYWNlcyk7XG4gICAgICAgICAgfX1cbiAgICAgICAgICB0eXBlPVwibGlua1wiXG4gICAgICAgID5cbiAgICAgICAgICB7d29ya3NwYWNlfVxuICAgICAgICA8L0J1dHRvbj5cbiAgICAgICAgPFN0eWxlZE1vZGFsXG4gICAgICAgICAgdGl0bGU9XCJXb3Jrc3BhY2VzXCJcbiAgICAgICAgICB2aXNpYmxlPXtvcGVuV29ya3NwYWNlc31cbiAgICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlV29ya3NwYWNlcyhmYWxzZSl9XG4gICAgICAgICAgZm9vdGVyPXtbXG4gICAgICAgICAgICA8U3R5bGVkQnV0dG9uXG4gICAgICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XG4gICAgICAgICAgICAgIGJhY2tncm91bmQ9XCJ3aGl0ZVwiXG4gICAgICAgICAgICAgIGtleT1cIkNsb3NlXCJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gdG9nZ2xlV29ya3NwYWNlcyhmYWxzZSl9XG4gICAgICAgICAgICA+XG4gICAgICAgICAgICAgIENsb3NlXG4gICAgICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXG4gICAgICAgICAgXX1cbiAgICAgICAgPlxuICAgICAgICAgIDxUYWJzIGRlZmF1bHRBY3RpdmVLZXk9XCIxXCIgdHlwZT1cImNhcmRcIj5cbiAgICAgICAgICAgIHt3b3Jrc3BhY2VzLm1hcCgod29ya3NwYWNlOiBXb3JzcGFjZVByb3BzKSA9PiAoXG4gICAgICAgICAgICAgIDxUYWJQYW5lIGtleT17d29ya3NwYWNlLmxhYmVsfSB0YWI9e3dvcmtzcGFjZS5sYWJlbH0+XG4gICAgICAgICAgICAgICAge3dvcmtzcGFjZS53b3Jrc3BhY2VzLm1hcCgoc3ViV29ya3NwYWNlOiBhbnkpID0+IChcbiAgICAgICAgICAgICAgICAgIDxCdXR0b25cbiAgICAgICAgICAgICAgICAgICAga2V5PXtzdWJXb3Jrc3BhY2UubGFiZWx9XG4gICAgICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcbiAgICAgICAgICAgICAgICAgICAgb25DbGljaz17YXN5bmMgKCkgPT4ge1xuICAgICAgICAgICAgICAgICAgICAgIHNldFdvcmtzcGFjZShzdWJXb3Jrc3BhY2UubGFiZWwpO1xuICAgICAgICAgICAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcbiAgICAgICAgICAgICAgICAgICAgICAvL2lmIHdvcmtzcGFjZSBpcyBzZWxlY3RlZCwgZm9sZGVyX3BhdGggaW4gcXVlcnkgaXMgc2V0IHRvICcnLiBUaGVuIHdlIGNhbiByZWdvbml6ZVxuICAgICAgICAgICAgICAgICAgICAgIC8vdGhhdCB3b3Jrc3BhY2UgaXMgc2VsZWN0ZWQsIGFuZCB3ZWUgbmVlZCB0byBmaWx0ZXIgdGhlIGZvcnN0IGxheWVyIG9mIGZvbGRlcnMuXG4gICAgICAgICAgICAgICAgICAgICAgYXdhaXQgc2V0V29ya3NwYWNlVG9RdWVyeShxdWVyeSwgc3ViV29ya3NwYWNlLmxhYmVsKTtcbiAgICAgICAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgICAgICAge3N1YldvcmtzcGFjZS5sYWJlbH1cbiAgICAgICAgICAgICAgICAgIDwvQnV0dG9uPlxuICAgICAgICAgICAgICAgICkpfVxuICAgICAgICAgICAgICA8L1RhYlBhbmU+XG4gICAgICAgICAgICApKX1cbiAgICAgICAgICA8L1RhYnM+XG4gICAgICAgIDwvU3R5bGVkTW9kYWw+XG4gICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgIDwvRm9ybT5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IFdvcmtzcGFjZXM7XG4iXSwic291cmNlUm9vdCI6IiJ9